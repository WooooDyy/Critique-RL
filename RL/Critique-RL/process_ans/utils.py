from process_ans.eval_utils import extract_answer, compare_answer_with_groundtruth

def compare_answer(answer: str, response: str, dataset_type: str, mode:str="cot") -> bool:
    if dataset_type == "omnimath":
        return True
    re_ans = extract_answer(response, dataset_name=dataset_type, mode=mode)
    gt_ans = extract_answer(answer, dataset_name=dataset_type, mode=mode)
    return compare_answer_with_groundtruth(re_ans, gt_ans)