from beam import Beam
from dataset import LineDataset
from fuzzywuzzy import fuzz
from clearml import Task, Logger
from utils import *
import pandas

logger = logging.getLogger(__name__)

# args, model, tokenizer, file_type='test', load_file="train", res_file="dense.pkl", before_code_file=""
def eval_line_completion(args, model, tokenizer, file_type='test', load_file="train", res_file="dense.pkl"):
    """
    Evaluate line level code completion on exact match and edit similarity.

    It is recommanded to use single GPU because it could not be batched.
    """

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
    # tokenizer, args, logger, file_type='test', block_size=924, load_file=None, search_res=None
    dataset = LineDataset(tokenizer, args, logger, file_type=file_type, block_size=1024 - 100,
                          load_file=load_file, search_res=res_file)
    test_sampler = SequentialSampler(dataset)
    test_dataloader = DataLoader(dataset, sampler=test_sampler, batch_size=1)
    model.to(args.device)
    # model.zero_grad()
    model.eval()

    def repackage_hidden(h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(repackage_hidden(v) for v in h)

    if args.langs == "python":
        break_ids = [tokenizer.sep_token_id]
    else:
        break_ids = [tokenizer.convert_tokens_to_ids('Ġ;'), tokenizer.convert_tokens_to_ids('Ġ}'),
                     tokenizer.convert_tokens_to_ids('Ġ{')]
    preds = []
    gts = []
    edit_sim = 0.0
    em = 0.0
    for step, (inputs, gt) in enumerate(test_dataloader):
        inputs = inputs.to(args.device)
        with torch.no_grad():
            beam_size = 5
            m = torch.nn.LogSoftmax(dim=-1)
            # This is a bug. which means that the last token is fed into the model.
            # model_outputs = model(inputs)
            model_outputs = model(inputs[:, :-1])

            outputs = model_outputs[1]
            p = []
            zero = torch.cuda.LongTensor(1).fill_(0)
            for i in range(inputs.shape[0]):
                if args.model_type == "rnn":
                    past_hidden = tuple(x[:, i:i + 1].expand(-1, beam_size, -1).contiguous() for x in outputs)
                else:
                    past = [torch.cat([x[0].unsqueeze(0), x[1].unsqueeze(0)], dim=0) if type(x) == tuple else x for x in
                            outputs]
                    past_hidden = [x[:, i:i + 1].expand(-1, beam_size, -1, -1, -1) for x in past]
                beam = Beam(beam_size, inputs[i][-1].cpu().data, break_ids)
                input_ids = None
                for _ in range(100):
                    if beam.done():
                        break
                    input_ids = beam.getCurrentState()
                    if args.model_type == "rnn":
                        outputs = model(input_ids, hidden=repackage_hidden(past_hidden))
                    else:
                        outputs = model(input_ids, past_key_values=past_hidden)

                    out = m(outputs[0][:, -1, :]).data
                    beam.advance(out)
                    if args.model_type == "rnn":
                        past_hidden = tuple(
                            x.data.index_select(1, beam.getCurrentOrigin()).contiguous() for x in outputs[1])
                    else:
                        past = [torch.cat([x[0].unsqueeze(0), x[1].unsqueeze(0)], dim=0) if type(x) == tuple else x for
                                x in outputs[1]]
                        past_hidden = [x.data.index_select(1, beam.getCurrentOrigin()) for x in past]
                hyp = beam.getHyp(beam.getFinal())
                pred = beam.buildTargetTokens(hyp)[:beam_size]

                pred = [torch.cat([x.view(-1) for x in p] + [zero] * (100 - len(p))).view(1, -1) for p in pred]
                p.append(torch.cat(pred, 0).unsqueeze(0))
            p = torch.cat(p, 0)
            for pred in p:
                t = pred[0].cpu().numpy()
                t = t.tolist()
                if 0 in t:
                    t = t[:t.index(0)]
                if args.langs == "python":
                    text = DecodeIds(t).strip("<EOL>").strip()
                else:
                    text = DecodeIds(t).strip("{").strip()
                #print(text)
                # exit()
                preds.append(text)
                gts.append(gt[0])
                edit_sim += fuzz.ratio(text, gt[0])
                em += 1 if text == gt[0] else 0
        if step % 100 == 0:
            logger.info(f"{step} are done!")
            #logger.info(f"EM: {edit_sim / len(preds)}")
            logger.info(f"Edit sim: {edit_sim / len(preds)}, EM: {em / len(preds)}")

    file_name = "prediction_line_reacc.txt"
    saved_file = os.path.join(args.output_dir, file_name)
    with open(saved_file, "w") as f:
        for i, (pred_text, gt) in enumerate(zip(preds, gts)):
            if pred_text == gt:
                label = 1
            else:
                label = 0
            save_json = {
                'label': label,
                'pred': pred_text,
                'gt': gt
            }

            f.write(json.dumps(save_json) + "\n")

    logger.info(f"Test {len(preds)} samples")
    logger.info(f"Edit sim: {edit_sim / len(preds)}, EM: {em / len(preds)}")

    result = {
        "Edit": float(edit_sim / len(preds)),
        "EM": float(em / len(preds))
    }

    output_eval_file = os.path.join(args.output_dir, "eval_line_result.txt")
    with open(output_eval_file, "w") as writer:
        # logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            # logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
    return result


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
        print("<!--- process train dataset --->")
        # 1. 为数据库切分代码
        split_file_path = split_code(args.dstore_file, args.dstore_path, max_chunk_len=300)
        # 3. 为token completion 处理数据
        #build_token_completion_data(generator_tokenizer, args, logger, file_type='test', block_size=1024)
    else:
        split_file_path = args.dstore_path + '/' + args.dstore_file.split("/")[-1].split(".")[0] + "_split.txt"

    before_contexts_file = os.path.join(args.data_dir, "test.json")

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

        result = eval_line_completion(args, generator_model, generator_tokenizer, file_type='test',
                                      load_file=load_file, res_file=res_file)

        print(result)


if __name__ == '__main__':
    main()