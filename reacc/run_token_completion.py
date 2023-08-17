from clearml import Task, Logger
from utils import *
import pandas

logger = logging.getLogger(__name__)


def evaluate(args, model, tokenizer, file_type='test', load_file="train", res_file="dense.pkl", before_code_file=""):
    logger.info("eval token-level completion.")

    model.to(args.device)

    if load_file is None:
        dataset = TokenCompletionDataset(tokenizer, args, logger, file_type=file_type,
                                         block_size=1024, before_code_file=before_code_file)
    else:
        dataset = TokenCompletionDataset(
            tokenizer, args, logger, file_type=file_type, block_size=1024,
            load_file=load_file,
            search_res=res_file,
            before_code_file=before_code_file
        )

    eval_batch_size = 8

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, collate_fn=partial(my_collect_fn, batch_first=True,
                                                                                   padding_value=tokenizer.pad_token_id),
                                 batch_size=eval_batch_size)

    logger.info("***** Running test *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", eval_batch_size)

    def DecodeIds(idxs):
        codes = ""
        for idx in idxs:
            to_add = tokenizer.convert_ids_to_tokens(idx)
            if tokenizer.convert_ids_to_tokens(idx)[0] == '\u0120':
                if not codes.endswith(" "):
                    codes += " " + to_add[1:]
                else:
                    codes += to_add[1:]
            elif (
                    idx in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id,
                            tokenizer.pad_token_id] or
                    tokenizer.convert_ids_to_tokens(idx).startswith("<NUM_LIT")
            ):
                codes += " " + to_add + " "
            else:
                codes += to_add
        return codes.strip(" ")

    model.eval()

    correct = 0.0
    total = 0

    total_pred = []
    total_gt = []

    start_time = time.time()
    #tokens = args.block_size * eval_dataloader.__len__()
    tmp_pred = []
    tmp_gt = []
    for step, (batch, token_labels) in enumerate(eval_dataloader):
        inputs = batch.to(args.device)
        token_labels = token_labels.to(args.device)

        with torch.no_grad():
            outputs = model(inputs)
            pred_scores = outputs[0]
            pred_ids = pred_scores.argmax(-1)

        all_pred = []
        all_gt = []
        prev_pred = None

        for pred, gt, token_label in zip(pred_ids, inputs, token_labels):
            pred = pred[token_label == 2]
            gt = gt[token_label == 2]
            # print(tokenizer.decode(gt))
            # print(gt)
            # print(all_gt)
            tmp_gt.extend(gt.cpu().tolist())
            tmp_pred.extend(pred.cpu().tolist())
            if tokenizer.eos_token_id in gt:
                pred = tmp_pred
                gt = tmp_gt
                for i, y in enumerate(gt):
                    if i == 0:
                        if y in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id,
                                 tokenizer.pad_token_id]:
                            now_gt = [y]
                            now_pred = [0] if prev_pred is None else [prev_pred]
                            all_pred.append(DecodeIds(now_pred).strip().split()[0])
                            all_gt.append(DecodeIds(now_gt).strip())
                            now_gt = []
                            now_pred = []
                        else:
                            now_gt = [y]
                            now_pred = [0] if prev_pred is None else [prev_pred]
                    else:
                        if tokenizer.convert_ids_to_tokens(y)[0] == '\u0120':
                            if len(now_gt) > 0:
                                cur_gt = DecodeIds(now_gt).strip().split()
                                try:
                                    cur_pred = DecodeIds(now_pred).strip().split()
                                    if len(cur_gt) <= len(cur_pred):
                                        cur_pred = cur_pred[:len(cur_gt)]
                                    else:
                                        pad_len = len(cur_gt) - len(cur_pred)
                                        cur_pred = cur_pred + ['SPACE'] * pad_len
                                    all_pred.extend(cur_pred)
                                except IndexError:
                                    all_pred.append("<SPACE>")
                                all_gt.extend(cur_gt)
                                now_gt = []
                                now_pred = []
                        if y in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id,
                                 tokenizer.pad_token_id] \
                                or tokenizer.convert_ids_to_tokens(y).startswith("<NUM_LIT") \
                                or tokenizer.convert_ids_to_tokens(y).startswith("<STR_LIT") \
                                or tokenizer.convert_ids_to_tokens(y).startswith("<CHAR_LIT"):
                            if len(now_gt) > 0:
                                cur_gt = DecodeIds(now_gt).strip().split()
                                try:
                                    cur_pred = DecodeIds(now_pred).strip().split()
                                    if len(cur_gt) <= len(cur_pred):
                                        cur_pred = cur_pred[:len(cur_gt)]
                                    else:
                                        pad_len = len(cur_gt) - len(cur_pred)
                                        cur_pred = cur_pred + ['SPACE'] * pad_len
                                    all_pred.extend(cur_pred)
                                except IndexError:
                                    all_pred.append("<SPACE>")
                                all_gt.extend(cur_gt)
                            now_gt = [y]
                            now_pred = [pred[i - 1]]
                            try:
                                all_pred.append(DecodeIds(now_pred).strip().split()[0])
                            except IndexError:
                                all_pred.append("<SPACE>")
                            all_gt.append(DecodeIds(now_gt).strip())
                            now_gt = []
                            now_pred = []
                            continue
                        now_gt.append(y)
                        now_pred.append(pred[i - 1])

                tmp_pred = []
                tmp_gt = []

        assert len(all_pred) == len(all_gt)

        total_pred.extend(all_pred)
        total_gt.extend(all_gt)

        for x, y in zip(all_pred, all_gt):
            if y not in ["<s>", "</s>", "<EOL>", "<pad>"]:
                total += 1
                if x == y:
                    correct += 1

        if step % 1000 == 0 and total > 0:
            logger.info(f"{step} are done!")
            logger.info(f"{total}, {correct / total}")

    end_time = time.time()

    print('Inference token per seconds', len(total_gt) / (end_time - start_time))

    prediction_name = args.model_type
    if args.use_dense:
        prediction_name += "__dense"
    if args.use_bm25:
        prediction_name += "__bm25"
    if args.use_hybrid:
        prediction_name += "__hybrid"
    prediction_name += "_predictions.txt"

    saved_file = os.path.join(args.output_dir, prediction_name)
    # preds, gts, true_gts, code_types, saved_file
    true_gts, skip_ids = read_true_gts(args.data_dir, file_type)
    total, result = post_process(args, total_pred, total_gt,
                                 true_gts,
                                 read_code_types(args.data_dir, file_type, skip_ids),
                                 saved_file)
    logger.info(f"Eval on {total} tokens, saved at {saved_file}")
    result['time'] = (end_time - start_time)

    return correct / total, result, total


def read_true_gts(data_dir, file_type):
    true_gts = []
    skip_ids = []
    data = open(os.path.join(data_dir, f"{file_type}.txt")).readlines()
    for id, s in enumerate(data):
        code = json.loads(s)
        tmp = len([t for t in code if '<STR_LIT' in t])
        if tmp > 1024:
            skip_ids.append(id)
            continue
        true_gts.append(code)

    print('true gts', len(true_gts))
    print('skip size', len(skip_ids))
    return true_gts, skip_ids


def read_code_types(data_dir, file_type, skip_ids):
    code_types = []
    data = open(os.path.join(data_dir, f"{file_type}_type.txt")).readlines()
    for id, s in enumerate(data):
        if id in skip_ids:
            continue
        code_type = json.loads(s)
        code_types.append(code_type)
    return code_types


def post_process(args, preds, gts, true_gts, code_types, saved_file):
    wf = open(saved_file, "w")

    cnt = 0
    new_gt = []
    new_pred = []

    total = 0
    correct = 0.0
    code_type_dict = {}
    code_type_correct = {}

    prev_gt = []
    idid = 0

    for i, (pred, gt) in enumerate(zip(preds, gts)):
        if gt in ["", "<pad>"]:
            continue
        new_gt.append(gt)
        new_pred.append(pred.replace(" ", ""))
        if gt == "</s>":
            gt_str = " ".join(new_gt)
            pred_str = " ".join(new_pred)
            true_gt = true_gts[cnt]
            true_gt_str = ' '.join(true_gt).strip()

            if gt_str != true_gt_str:
                with open('t.txt', 'w') as f:
                    f.write(gt_str + '\n')
                    f.write(true_gt_str)
                print(prev_gt)
                print(idid)
                # print('skip!!!')
                # cnt += 1
                # continue
            assert gt_str == true_gt_str, f"{cnt} sample gt_str != true_gt"
            wf.write(pred_str + "\n")

            code_type = code_types[cnt]
            assert len(new_gt) == len(code_type)

            for j, (x, y, z) in enumerate(zip(new_pred, new_gt, code_type)):
                if y not in ["<s>", "</s>", "<EOL>", "<pad>"] and z != 'unknown':
                    total += 1
                    if z not in code_type_dict:
                        code_type_dict[z] = 0
                        code_type_correct[z] = 0
                    code_type_dict[z] += 1
                    if x == y:
                        correct += 1
                        code_type_correct[z] += 1

            cnt += 1

            prev_gt = gt_str
            idid = i

            new_gt = []
            new_pred = []

    code_type_correct = {k: round(v / code_type_dict[k] * 100, 2) for k, v in code_type_correct.items()}
    code_type_dict = {k: round(v / total * 100, 2) for k, v in code_type_dict.items()}

    logger.info(f"Total {total} tokens, accuracy: {round(correct / total * 100, 2)}")
    logger.info(f"Percent code types: " + json.dumps(code_type_dict))
    logger.info(f"Code type accuracy: " + json.dumps(code_type_correct))

    text = args.output_dir + '\n'
    table_name = ''
    table_per = ''
    table_val = ''
    for k in code_type_dict.keys():
        table_name += k + '\t'
        table_per += str(code_type_dict[k]) + '\t'
        table_val += str(code_type_correct[k]) + '\t'
    table_val += str({round(correct / total * 100, 2)}) + '\t'
    text += table_name + '\n' + table_per + '\n' + table_val + '\n'

    result = {}
    for k in code_type_dict.keys():
        result[k + "_percent"] = [code_type_dict[k]]
        result[k] = [code_type_correct[k]]
    result['avg'] = [round(correct / total * 100, 2)]

    file_name = ''
    if args.model_type == "unixCoder":
        if args.use_bm25:
            file_name = f'unixCoder_bm25.txt'
        elif args.use_dense:
            file_name = f'unixCoder_dense.txt'
        else:
            file_name = 'unixCoder_hybrid.txt'
    else:
        if args.use_bm25:
            file_name = f'gpt_bm25.txt'
        elif args.use_dense:
            file_name = f'gpt_dense.txt'
        else:
            file_name = 'gpt_hybrid.txt'
    with open(file_name, 'a+') as f:
        f.write(text)

    return total, result


def add_args(parser):
    # 数据库相关
    parser.add_argument("--dstore_file", default=None, type=str, required=True,
                        help="The datastore file path. [domain training file]")

    # 预训练模型相关
    parser.add_argument("--model_type", default="gpt2", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--pretrain_dir", default="", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--lit_file", type=str,
                        help="literals json file")

    # 待补全文件相关
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data path.")
    parser.add_argument("--langs", default=None, type=str, required=True,
                        help="Languages to train, if all, train all languages in data_dir")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # 命令相关
    parser.add_argument("--data_process", action="store_true")
    parser.add_argument("--build_index", action='store_true')
    parser.add_argument("--do_search", action='store_true')
    parser.add_argument("--do_generate", action='store_true')
    parser.add_argument("--use_dense", action='store_true')
    parser.add_argument("--use_bm25", action='store_true')
    parser.add_argument("--use_hybrid", action='store_true')
    parser.add_argument("--bm_name", default="bm25", type=str, required=False,
                        help="elasticsearch name.")
    parser.add_argument('--clearml_proj_name', type=str, default='')

def main():
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.bm_name = args.model_type.lower()  # elasticsearch needs lower index

    args.dstore_path = args.output_dir + '/datastore'

    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)

    # 为了测试需要，需要删除所有文件
    if not os.path.exists(args.dstore_path):
        os.makedirs(args.dstore_path)

    description = args.pretrain_dir

    if args.use_bm25:
        description += "__bm25"
    if args.use_hybrid:
        description += "__hybrid"
    if args.use_dense:
        description += "__dense"

    Task.init(project_name=args.clearml_proj_name, task_name=args.data_dir + "__" + description)

    # setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    # 使用FileHandler输出到文件
    fh = logging.FileHandler("log.log")
    logger.addHandler(fh)

    # get special tokens
    special_tokens = get_special_tokens(args.lit_file)

    # 加载retriever
    retrieve_tokenizer, retrieve_model = load_retriever()

    # 加载generator
    generator_tokenizer, generator_model = load_pretrained_model(args, special_tokens)

    if args.data_process:
        print("<!--- process train and test dataset --->")
        # 1. 为数据库切分代码
        split_file_path = split_code(args.dstore_file, args.dstore_path, max_chunk_len=300)
        # 3. 为token completion 处理数据
        build_token_completion_data(generator_tokenizer, args, logger, file_type='test', block_size=1024)
    else:
        split_file_path = args.dstore_path + '/' + args.dstore_file.split("/")[-1].split(".")[0] + "_split.txt"

    before_contexts_file = args.dstore_path + "/test_query.json"

    # time 包含三部分: 搜索 + 推断
    time_search = 0

    if args.build_index:
        print("<!--- build index --->")
        # 2. 为数据库存储向量
        index_file = save_vec(args, split_file_path, retrieve_tokenizer, retrieve_model, 'dstore_keys',
                              output_path=args.dstore_path, lang=args.langs, api=True)

        print("<!--- build query --->")
        # 4. 为token completion 创建query向量
        start_time = time.time()
        query_file = save_vec(args, before_contexts_file, retrieve_tokenizer, retrieve_model, 'dstore_queries',
                              output_path=args.dstore_path, lang=args.langs, api=True)
        end_time = time.time()

        time_search += end_time - start_time

    if args.do_search:
        start_time = time.time()
        if args.use_dense:
            index_file_path = args.dstore_path + "/dstore_keys.pkl"
            query_file_path = args.dstore_path + "/dstore_queries.pkl"
            logger.info('<!-- do dense search -->')
            file_path = args.dstore_path + "/dense.pkl"
            search_dense(index_file_path, query_file_path, file_path)
            save_path = args.dstore_path + "/dense_res.pkl"
            get_res(bm25_file="", dense_file=file_path, save_file=save_path, alpha=0.9)

        if args.use_bm25:
            logger.info('<!-- do bm25 search -->')
            tmp_dir = args.dstore_path + "/tmp"
            file_path = args.dstore_path + "/bm25.pkl"
            search_bm25(split_file_path, before_contexts_file, tmp_dir, args.bm_name, file_path)
            save_path = args.dstore_path + "/bm25_res.pkl"
            get_res(bm25_file=file_path, dense_file="", save_file=save_path, alpha=0.9)

        if args.use_hybrid:
            logger.info('<!-- do hybrid search -->')
            bm25_file_path = args.dstore_path + "/bm25.pkl"
            dense_file_path = args.dstore_path + "/dense.pkl"
            save_path = args.dstore_path + "/hybrid_res.pkl"
            get_res(bm25_file=bm25_file_path, dense_file=dense_file_path,
                    save_file=save_path, alpha=0.9)

        end_time = time.time()

        time_search += end_time - start_time

    if args.do_generate:
        logger.info('<!-- do generate -->')
        load_file = "train_split" if args.use_dense or args.use_bm25 else None
        if args.use_dense:
            res_file = "dense_res"
        elif args.use_bm25:
            res_file = "bm25_res"
        elif args.use_hybrid:
            res_file = "hybrid_res"
        else:
            res_file = None

        word_acc, result, total_tokens = evaluate(args, generator_model, generator_tokenizer, file_type='test',
                                                  load_file=load_file, res_file=res_file,
                                                  before_code_file="test_query.json")

        logger.info("word acc: " + str(word_acc))

        result['intra_project'] = [args.data_dir]
        result['description'] = [description]
        db_size = get_dir_size(args.dstore_path)
        result['db_size'] = db_size
        result['time'] = total_tokens / (result['time'] + time_search)

        df = pandas.DataFrame(result)
        df.to_csv(args.model_type + '_domain_reacc_result.csv', mode='a')
        Logger.current_logger().report_table(
            "Project with knn",
            "PD with index",
            table_plot=df
        )


if __name__ == '__main__':
    main()



