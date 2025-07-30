import json
import re
from math import isclose
from collections import defaultdict

with open("", "r", encoding="utf-8") as file2:
    data_answer = json.load(file2)

print(len(data_answer))  


def eval_acc_and_f1(samples):
    evaluated_samples = [sample for sample in samples if "score" in sample]
    if not evaluated_samples:
        return 0.0, 0.0
    
    acc = sum([sample["score"] for sample in evaluated_samples])/len(evaluated_samples)
    try:
        recall = sum([sample["score"] for sample in evaluated_samples if sample["answer"]!="Not answerable"])/len([sample for sample in evaluated_samples if sample["answer"]!="Not answerable"])
        precision = sum([sample["score"] for sample in evaluated_samples if sample["answer"]!="Not answerable"])/len([sample for sample in evaluated_samples if sample["pred"]!="Not answerable"])
        f1 = 2*recall*precision/(recall+precision) if (recall+precision)>0.0 else 0.0
    except:
        f1 = 0.0
    
    return acc, f1


def show_results(samples, show_path=""):
    for sample in samples:
        sample["evidence_pages"] = eval(sample["evidence_pages"])
        sample["evidence_sources"] = eval(sample["evidence_sources"])
    
    with open(show_path, 'w') as f:
        acc, f1 = eval_acc_and_f1(samples)
        f.write("Overall Acc: {} | Question Number: {}\n".format(acc, len(samples)))
        f.write("Overall F1-score: {} | Question Number: {}\n".format(f1, len(samples)))
        f.write("-----------------------\n")

        #####################
        acc_single_page, _ = eval_acc_and_f1([sample for sample in samples if len(sample["evidence_pages"])==1])
        acc_multi_page, _ = eval_acc_and_f1([sample for sample in samples if len(sample["evidence_pages"])!=1 and sample["answer"]!="Not answerable"])
        acc_neg, _ = eval_acc_and_f1([sample for sample in samples if sample["answer"]=="Not answerable"])

        f.write("Single-page | Accuracy: {} | Question Number: {}\n".format(
            acc_single_page, len([sample for sample in samples if len(sample["evidence_pages"])==1])
        ))
        f.write("Cross-page | Accuracy: {} | Question Number: {}\n".format(
            acc_multi_page, len([sample for sample in samples if len(sample["evidence_pages"])!=1 and sample["answer"]!="Not answerable"])
        ))
        f.write("Unanswerable | Accuracy: {} | Question Number: {}\n".format(
            acc_neg, len([sample for sample in samples if sample["answer"]=="Not answerable"])
        ))
        f.write("-----------------------\n")

        #####################
        source_sample_dict, document_type_dict = defaultdict(list), defaultdict(list)
        for sample in samples:
            for answer_source in sample["evidence_sources"]:
                source_sample_dict[answer_source].append(sample)
            document_type_dict[sample["doc_type"]].append(sample)
        for type, sub_samples in source_sample_dict.items():
            f.write(
                "Evidence Sources: {} | Accuracy: {} | Question Number: {}\n".format(type, eval_acc_and_f1(sub_samples)[0], len(sub_samples))
            )

        f.write("-----------------------\n")
        for type, sub_samples in document_type_dict.items():
            f.write(
                "Document Type: {} | Accuracy: {} | Question Number: {}\n".format(type, eval_acc_and_f1(sub_samples)[0], len(sub_samples))
            )

samples = data_answer
show_results(samples)