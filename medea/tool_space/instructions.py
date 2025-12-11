# Task Instructions
task_instructions = {
    "claim_no_context": ("Given a scientific claim, answer if the scientific claim is factually correct (true) or not (false). For each scientific claim provided, simply state whether it is true or false. If the statement is supported by the paragraph, answer true; otherwise answer false. You don't need to provide any explanation, just the label.", "\nClaim: "), 
    "claim_gold": ("Given a scientific claim and a gold paragraph that may support or contradict with the claim, answer if the scientific claim is factually correct or not. For each scientific claim provided, simply state whether it is true or false. If the statement is supported by the paragraph, answer true; otherwise answer false. You don't need to provide any explanation, just the label.", "\nClaim: "),
    "claim_full": ("Given a scientific claim and a set of relevant paragraphs, that may support or contradict with the claim, answer if the scientific claim is factually correct or not. For each scientific claim provided, simply state whether it is true or false. If the statement is supported by the paragraph, answer true; otherwise answer false. You don't need to provide any explanation, just the label. You also need to provide the citation numbers that support your answer. Your citation is presented as [i], where i corresponds to the number in the 'References: '.", "\nClaim: "),
    "boolean_question_no_context": ("Given a question related to scientific literature, answer yes or no. Simply state whether it is yes or no. You don't need to provide any explanation, just the label.", "\nQuestion: "),
    "boolean_question_gold": ("Given a question related to scientific literature and a gold paragraph that provides sufficient information to answer the question, answer yes or no. Simply state whether it is yes or no.","\nQuestion:" ),
    "boolean_question_full": ("Given a question related to scientific literature and a set of reference passages that may provide sufficient information to answer the question, answer yes or no. Simply state whether it is yes or no. You don't need to provide any explanation, just the label. You also need to provide the citation numbers that support your answer. Your citation is presented as [i], where i corresponds to the number in the 'References: '.", "\nQuestion: "),
}

demonstrations = {
    "claim_no_context": "Your answer must be marked by special tokens, [Response_Start] and [Response_End]. For example, the input and output looks like this:\nClaim: 1 in 5 million in UK have abnormal PrP positivity.\n[Response_Start]false[Response_End]\nNow please verify the following claim.", 
    "claim_gold": "Your answer must be marked by special tokens, [Response_Start] and [Response_End]. For example, the input and output looks like this: \nReferences: \n[0] Title: Prevalent abnormal prion protein in human appendixes after bovine spongiform encephalopathy epizootic: large scale survey Text: OBJECTIVES To carry out a further survey of archived appendix samples to understand better the differences between existing estimates of the prevalence of subclinical infection with prions after the bovine spongiform encephalopathy epizootic and to see whether a broader birth cohort was affected, and to understand better the implications for the management of blood and blood products and for the handling of surgical instruments. DESIGN Irreversibly unlinked and anonymised large scale survey of archived appendix samples. SETTING Archived appendix samples from the pathology departments of 41 UK hospitals participating in the earlier survey, and additional hospitals in regions with lower levels of participation in that survey. SAMPLE 32,441 archived appendix samples fixed in formalin and embedded in paraffin and tested for the presence of abnormal prion protein (PrP). RESULTS Of the 32,441 appendix samples 16 were positive for abnormal PrP, indicating an overall prevalence of 493 per million population (95% confidence interval 282 to 801 per million). The prevalence in those born in 1941-60 (733 per million, 269 to 1596 per million) did not differ significantly from those born between 1961 and 1985 (412 per million, 198 to 758 per million) and was similar in both sexes and across the three broad geographical areas sampled. Genetic testing of the positive specimens for the genotype at PRNP codon 129 revealed a high proportion that were valine homozygous compared with the frequency in the normal population, and in stark contrast with confirmed clinical cases of vCJD, all of which were methionine homozygous at PRNP codon 129. CONCLUSIONS This study corroborates previous studies and suggests a high prevalence of infection with abnormal PrP, indicating vCJD carrier status in the population compared with the 177 vCJD cases to date. These findings have important implications for the management of blood and blood products and for the handling of surgical instruments.\nClaim: 1 in 5 million in UK have abnormal PrP positivity. \n[Response_Start]false[Response_End]\nNow please verify the following claim.\n",
    "claim_full": """
    Your answer must be marked by special tokens, [Response_Start] and [Response_End]. For example, the input and output looks like this: 
    References: 
    [0] Title: MLQA: Evaluating Cross-lingual Extractive Question Answering Text: Question answering (QA) models have shown rapid progress enabled by the availability of large, high-quality benchmark datasets. Such annotated datasets are difficult and costly to collect, and rarely exist in languages other than English, making building QA systems that work well in other languages challenging. In order to develop such systems, it is crucial to invest in high quality multilingual evaluation benchmarks to measure progress. We present MLQA, a multi-way aligned extractive QA evaluation benchmark intended to spur research in this area. MLQA contains QA instances in 7 languages, English, Arabic, German, Spanish, Hindi, Vietnamese and Simplified Chinese. MLQA has over 12K instances in English and 5K in each other language, with each instance parallel between 4 languages on average. We evaluate state-of-the-art cross-lingual models and machine-translation-based baselines on MLQA. In all cases, transfer results are shown to be significantly behind training-language performance.
    [1] Title: XOR QA: Cross-lingual Open-Retrieval Question Answering Text: Multilingual question answering tasks typically assume that answers exist in the same language as the question. Yet in practice, many languages face both information scarcity—where languages have few reference articles—and information asymmetry—where questions reference concepts from other cultures. This work extends open-retrieval question answering to a cross-lingual setting enabling questions from one language to be answered via answer content from another language. We construct a large-scale dataset built on 40K information-seeking questions across 7 diverse non-English languages that TyDi QA could not find same-language answers for. Based on this dataset, we introduce a task framework, called Cross-lingual Open-Retrieval Question Answering (XOR QA), that consists of three new tasks involving cross-lingual document retrieval from multilingual and English resources. We establish baselines with state-of-the-art machine translation systems and cross-lingual pretrained models. Experimental results suggest that XOR QA is a challenging task that will facilitate the development of novel techniques for multilingual question answering.
    [2] Title: Unsupervised Cross-lingual Representation Learning at Scale Text: This paper shows that pretraining multilingual language models at scale leads to significant performance gains for a wide range of cross-lingual transfer tasks. We train a Transformer-based masked language model on one hundred languages, using more than two terabytes of filtered CommonCrawl data. Our model, dubbed XLM-R, significantly outperforms multilingual BERT (mBERT) on a variety of cross-lingual benchmarks, including +14.6% average accuracy on XNLI, +13% average F1 score on MLQA, and +2.4% F1 score on NER. XLM-R performs particularly well on low-resource languages, improving 15.7% in XNLI accuracy for Swahili and 11.4% for Urdu over previous XLM models. We also present a detailed empirical analysis of the key factors that are required to achieve these gains, including the trade-offs between (1) positive transfer and capacity dilution and (2) the performance of high and low resource languages at scale. Finally, we show, for the first time, the possibility of multilingual modeling without sacrificing per-language performance; XLM-R is very competitive with strong monolingual models on the GLUE and XNLI benchmarks. We will make our code, data and models publicly available.
    Claim: The XOR QA dataset covers eight languages. 
    [Response_Start]false [1][Response_End]
    Now please verify the following claim.\n 
    """,
    "boolean_question_no_context": "Your answer must be marked by special tokens, [Response_Start] and [Response_End]. For example, the input and output looks like this:\nQuestion: Did Chile's traffic law reform push police enforcement?\n[Response_Start]yes[Response_End]\nNow answer the following question.",
    "boolean_question_gold": "Your answer must be marked by special tokens, [Response_Start] and [Response_End]. For example, the input and output looks like this: \nReferences: \n[0] The objective of the current study is to determine to what extent the reduction of Chile's traffic fatalities and injuries during 2000-2012 was related to the police traffic enforcement increment registered after the introduction of its 2005 traffic law reform. A unique dataset with assembled information from public institutions and analyses based on ordinary least square and robust random effects models was carried out. Dependent variables were traffic fatality and severe injury rates per population and vehicle fleet. Independent variables were: (1) presence of new national traffic law; (2) police officers per population; (3) number of traffic tickets per police officer; and (4) interaction effect of number of traffic tickets per police officer with traffic law reform. Oil prices, alcohol consumption, proportion of male population 15-24 years old, unemployment, road infrastructure investment, years' effects and regions' effects represented control variables. Empirical estimates from instrumental variables suggest that the enactment of the traffic law reform in interaction with number of traffic tickets per police officer is significantly associated with a decrease of 8% in traffic fatalities and 7% in severe injuries. Piecewise regression model results for the 2007-2012 period suggest that police traffic enforcement reduced traffic fatalities by 59% and severe injuries by 37%. \nQuestion: Did Chile's traffic law reform push police enforcement?\n[Response_Start]yes[Response_End]\nNow answer the following question. ",
    "boolean_question_full": """
    Your answer must be marked by special tokens, [Response_Start] and [Response_End]. For example, the input and output looks like this: 
    References: 
    [0] The gap between evidence-based treatments and routine care has been well established. Findings from the Sequenced Treatments Alternatives to Relieve Depression (STAR*D) emphasized the importance of measurement-based care for the treatment of depression as a key ingredient for achieving response and remission; yet measurement-based care approaches are not commonly used in clinical practice. The Nine-Item Patient Health Questionnaire (PHQ-9) for monitoring depression severity was introduced in 19 diverse psychiatric practices. During the one-year course of the project the helpfulness and feasibility of implementation of PHQ-9 in these psychiatric practices were studied. The project was modeled after the Institute for Healthcare Improvement Breakthrough Series. Two of the 19 practices dropped out during the course of the project. By the conclusion of the study, all remaining 17 practices had adopted PHQ-9 as a routine part of depression care in their practice. On the basis of responses from 17 psychiatrists from those practices, PHQ-9 scores influenced clinical decision making for 93% of 6,096 patient contacts. With the additional information gained from the PHQ-9 score, one or more treatment changes occurred during 40% of these clinical contacts. Changing the dosage of antidepressant medication and adding another medication were the most common treatment changes recorded by psychiatrists, followed by starting or increasing psychotherapy and by switching or initiating antidepressants. In 3% of the patient contacts, using the PHQ-9 led to additional suicide risk assessment. 
    [1] To compare maternal and neonatal outcomes among grandmultiparous women to those of multiparous women 30 years or older. A database of the vast majority of maternal and newborn hospital discharge records linked to birth/death certificates was queried to obtain information on all multiparous women with a singleton delivery in the state of California from January 1, 1997 through December 31, 1998. Maternal and neonatal pregnancy outcomes of grandmultiparous women were compared to multiparous women who were 30 years or older at the time of their last birth. The study population included 25,512 grandmultiparous and 265,060 multiparous women 30 years or older as controls. Grandmultiparous women were predominantly Hispanic (56%). After controlling for potential confounding factors, grandmultiparous women were at significantly higher risk for abruptio placentae (odds ratio OR: 1.3; 95% confidence intervals CI: 1.2-1.5), preterm delivery (OR: 1.3; 95% CI: 1.2-1.4), fetal macrosomia (OR: 1.5; 95% CI: 1.4-1.6), neonatal death (OR: 1.5; 95% CI: 1.3-1.8), postpartum hemorrhage (OR: 1.2; 95% CI: 1.1-1.3) and blood transfusion (OR: 1.5; 95% CI: 1.3-1.8).', 'long_answer': 'Grandmultiparous women had increased maternal and neonatal morbidity, and neonatal mortality even after controlling for confounders, suggesting a need for closer observation than regular multiparous patients during labor and delivery.
    [2] The objective of the current study is to determine to what extent the reduction of Chile's traffic fatalities and injuries during 2000-2012 was related to the police traffic enforcement increment registered after the introduction of its 2005 traffic law reform. A unique dataset with assembled information from public institutions and analyses based on ordinary least square and robust random effects models was carried out. Dependent variables were traffic fatality and severe injury rates per population and vehicle fleet. Independent variables were: (1) presence of new national traffic law; (2) police officers per population; (3) number of traffic tickets per police officer; and (4) interaction effect of number of traffic tickets per police officer with traffic law reform. Oil prices, alcohol consumption, proportion of male population 15-24 years old, unemployment, road infrastructure investment, years' effects and regions' effects represented control variables. Empirical estimates from instrumental variables suggest that the enactment of the traffic law reform in interaction with number of traffic tickets per police officer is significantly associated with a decrease of 8% in traffic fatalities and 7% in severe injuries. Piecewise regression model results for the 2007-2012 period suggest that police traffic enforcement reduced traffic fatalities by 59% and severe injuries by 37%. 
    Question: Did Chile's traffic law reform push police enforcement?
    [Response_Start]yes [2][Response_End]
    Now answer the following question. 
    """,
}

# Examples
example_passages_rag = """
[0] Title: Attributed Question Answering: Evaluation and Modeling for Attributed Large Language Models Text: Roberts et al. (2020) shows that T5 (Raffel et al., 2020) can perform a new task formulation, closedbook QA. Concretely, T5 can produce answers to questions without access to any corpus at inference time, instead producing answers based on its model parameters, tuned to remember information digested in pretraining.\n
[1] Title: Reliable, Adaptable, and Attributable Language Models with Retrieval Text: Unlike parametric LMs—which use large-scale text data only during training—retrieval-augmented LMs leverage an external large-scale collection of documents (datastore) at inference by selecting relevant documents from the datastore (Asai et al., 2023a). Retrieval-augmented LMs can W1: largely reduce factual errors (Mallen et al., 2023), W2: provide better attributions (Gao et al., 2023a), W3: enabling flexible opt-in and out of sequences (Min et al., 2024).
[2] Title: Atlas: Few-shot Learning with Retrieval Augmented Language Models Text: In this work we present Atlas, a carefully designed and pre-trained retrieval augmented language model able to learn knowledge intensive tasks with very few training examples. We perform evaluations on a wide range of tasks, including MMLU, KILT and NaturalQuestions, and study the impact of the content of the document index, showing that it can easily be updated. Notably, Atlas reaches over 42% accuracy on Natural Questions using only 64 examples, outperforming a 540B parameters model by 3% despite having 50x fewer parameters.
[3] Title: Language Models are Few-Shot Learners Text: Similarly, GPT-3 achieves 64.3% accuracy on TriviaQA in the zero-shot setting, 68.0% in the one-shot setting, and 71.2% in the few-shot setting, the last of which is state-of-the-art relative to fine-tuned models operating in the same closed-book setting.
[4] Title: When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories Text:  On both datasets, LMs' memorization (RQ1) is often limited to the popular factual knowledge and even GPT-3 davinci-003 fails to answer the majority of the long-tail questions. Moreover, on such questions, scaling up models does not significantly improve the performance. This also suggests that we can predict if LMs memorize certain knowledge based on the information presented in the input question only. We next investigate whether a semi-parametric approach that augments LMs with retrieved evidence can mitigate the low performance on questions about less popular entities (RQ2). Nonparametric memories largely improve performance on long-tail distributions across models.
[5] Title: Democratizing Large Language Models via Personalized Parameter-Efficient Fine-tuning Text: Personalization in large language models (LLMs) is increasingly important, aiming to align LLM's interactions, content, and recommendations with individual user preferences. Recent advances in LLM personalization have spotlighted effective prompt design, by enriching user queries with non-parametric knowledge through behavior history retrieval and textual profiles. However, these approaches were limited due to a lack of model ownership, resulting in constrained customization and privacy issues. Moreover, they often failed to accurately capture user behavior patterns, especially in cases where user data were complex and dynamic. To address these shortcomings, we introduce One PEFT Per User (OPPU), which employs personalized parameter-efficient fine-tuning (PEFT) modules, to store user-specific behavior patterns and preferences.
[6] Title: RECOMP: Improving Retrieval-Augmented LMs with Context Compression and Selective Augmentation Text:  Retrieval-augmented language models (RALMs) (Khandelwal et al., 2019; Izacard et al., 2022; Lewis et al., 2020; Borgeaud et al., 2022) have shown impressive performance on knowledge-intensive tasks (Kwiatkowski et al., 2019; Petroni et al., 2021). Simply prepending retrieved documents to the input without updating the language models (LMs) (Shi et al., 2023b; Ram et al., 2023; Si et al., 2022) allows retrieval augmentation even for black-box LMs, but such approach comes with limitations. First, it increases computational costs as LMs now encode substantially more tokens. Second, even if we manage to adapt LMs to efficiently incorporate longer context (Beltagy et al., 2020; Zaheer et al., 2020), these models struggle to use all information in the context, frequently missing information placed in the middle (Liu et al., 2023). Third, prepending a large number of documents in-context can further confuse LMs with irrelevant information, degrading model performances (Mallen et al., 2022; Shi et al., 2023a).
"""
example_question_rag = "How do language models leverage parametric and non-parametric knowledge?"
example_answer_rag = """
Language models leverage both parametric and non-parametric knowledge to perform various tasks.\n
Parametric knowledge refers to the information stored in the model's parameters, which are learned during training [0]. This type of knowledge allows language models to perform tasks such as closed-book question answering, where the model produces answers based on its internal knowledge without accessing any external corpus [0]. However, language models' memorization of parametric knowledge is often limited to popular factual knowledge, and even large models like GPT-3 may fail to answer the majority of long-tail questions [4].\n
On the other hand, non-parametric knowledge is retrieved from an external source, such as a large-scale collection of documents, during inference [1]. This type of knowledge is used in retrieval-augmented language models, which can reduce factual errors, provide better attributions, and enable flexible opt-in and out of sequences [1]. Retrieval-augmented language models have been shown to be effective in few-shot learning scenarios, where they can learn knowledge-intensive tasks with very few training examples [2]. For example, the Atlas model, a retrieval-augmented language model, can reach over 42% accuracy on Natural Questions using only 64 examples, outperforming a 540B parameters model by 3% despite having 50x fewer parameters [2]. Moreover, even without training, simply combining off-the-shelf LMs such as GPT3 with retrieval augmentation can significantly improve performance in long-tail and have been shown to mitigate the low performance on questions about less popular entities[4]. However, retrieval-augmented LMs have several limitations. Specifically, retrieval-augmented LMs can make inference much more inefficient due to increased context length [6].\n
"""
example_answer_rag_incorrect = """
Language models leverage both parametric and non-parametric knowledge to perform various tasks. Parametric knowledge refers to the information stored in the model's parameters, which are learned during training [0]. This type of knowledge allows language models to perform tasks such as closed-book question answering, where the model produces answers based on its internal knowledge without accessing any external corpus [0]. However, language models' memorization of parametric knowledge is often limited to popular factual knowledge, and even large models like GPT-4 often fail to answer the majority of long-tail questions [4].\n
On the other hand, non-parametric knowledge is retrieved from an external source, such as a large-scale collection of documents, during inference [1]. This type of knowledge is used in retrieval-augmented language models, which can reduce factual errors, provide better attributions, and enable flexible opt-in and out of sequences [1]. Retrieval-augmented language models have been shown to be effective in few-shot learning scenarios, where they can learn knowledge-intensive tasks with very few training examples [2]. For example, the Atlas model, a retrieval-augmented language model, can reach over 42% accuracy on Natural Questions using only 64 examples, outperforming a 540B parameters model by 3% despite having 50x fewer parameters [2]. Moreover, even without training, simply combining off-the-shelf LMs such as GPT3 with retrieval augmentation can significantly improve performance in long-tail and have been shown to mitigate the low performance on questions about less popular entities [4]. However, retrieval-augmented LMs have several limitations. Specifically, retrieval-augmented LMs can make inference much more inefficient due to increased context length [6].\n
"""

example_feedback = """
Feedback: Only concrete examples used in the answer are QA results. We should include more results from non QA tasks. Question: What tasks retrieval-augmented LMs have been applied to?\n
Feedback: Only one limitation discussed in the answer is efficiency. Question: What are the disadvantages of retrieval-augmented LMs?\n
Feedback: The original answer can be improved by adding more logical structure e.g., grouping similar discussions together and add paragraph headers.\n
"""
example_question_peft = "Discuss various parameter-efficient fine-tuning (PEFT) techniques for large language models, highlighting their strengths and weaknesses."
example_passages_peft = """
[0] Title: Empirical Analysis of the Strengths and Weaknesses of PEFT Techniques for LLMs Text: As foundation models continue to exponentially scale in size, efficient methods of adaptation become increasingly critical. Parameter-efficient fine-tuning (PEFT), a recent class of techniques that require only modifying a small percentage of the model parameters, is currently the most popular method for adapting large language models (LLMs). Several PEFT techniques have recently been proposed with varying tradeoffs. We provide a comprehensive and uniform benchmark of various PEFT techniques across a representative LLM, the FLAN-T5 model, and evaluate model performance across different data scales of classification and generation datasets. Based on this, we provide a framework for choosing the optimal fine-tuning techniques given the task type and data availability. Contrary to popular belief, we also empirically prove that PEFT techniques converge slower than full tuning in low data scenarios, and posit the amount of data required for PEFT methods to both perform well and converge efficiently.\n
[1] Title: Prefix-Tuning: Optimizing Continuous Prompts for Generation Text: In this paper, we propose prefix-tuning, a lightweight alternative to fine-tuning for natural language generation tasks, which keeps language model parameters frozen, but optimizes a small continuous task-specific vector (called the prefix). Prefix-tuning draws inspiration from prompting, allowing subsequent tokens to attend to this prefix as if it were "virtual tokens". We apply prefix-tuning to GPT-2 for table-to-text generation and to BART for summarization. We find that by learning only 0.1\% of the parameters, prefix-tuning obtains comparable performance in the full data setting, outperforms fine-tuning in low-data settings, and extrapolates better to examples with topics unseen during training.\n
[2] Title: Parameter-Efficient Fine-Tuning for Pre-Trained Vision Models: A Survey Text: This paper aims to provide a comprehensive and systematic study of PEFT methods in the vision domain, particularly focusing on transformer-based pre-trained models ranging from the year 2019 to the year 2023. As shown in Fig. 1, existing visual PEFT methods could be categorized into addition-based tuning, partial-based tuning, and unified-based tuning. In section 2, we will define the problem of PEFT, introduce popular backbones, and discuss pre-training methods. In section 3, a detailed taxonomy and in-depth analysis of the PEFT methods will be presented. The real-world applications of PEFT will be introduced in section 4. Finally, in section 5, we will point out future research challenges.\n
[3] Title: Towards a Unified View of Parameter-Efficient Transfer Learning Text: To mitigate this issue, a few lightweight alternatives have been proposed to update only a small number of extra parameters while keeping most pretrained parameters frozen. For example, adapter tuning (Houlsby et al., 2019) inserts small neural modules called adapters to each layer of the pretrained network and only the adapters are trained at fine-tuning time. Inspired by the success of prompting methods that control PLMs through textual prompts (Brown et al., 2020; Liu et al., 2021a), prefix tuning (Li & Liang, 2021) and prompt tuning (Lester et al., 2021) prepend an additional l tunable prefix tokens to the input or hidden layers and only train these soft prompts when fine-tuning on downstream tasks.\n
[4] Title: I-Tuning: Tuning Frozen Language Models with Image for Lightweight Image Captioning  Text: We design an I-Tuning module to connect the pre-trained vision encoder (i.e., CLIP-ViT [7]) and the language decoder (i.e., GPT2 [8]). To align between the language and vision modals, it serves as a cross-modal filter that automatically picks the visual information from the output of the vision encoder and adjusts the output hidden states of the language decoder. During training, we only update the newly introduced parameters in the I-Tuning module, and the parameters of the two pre-trained models are frozen.\n
"""
example_rating_peft = """
[Response_Start][0] Rating: 3 Explanation: This paragraph discusses a high-level overview and goal of parameter efficient tuning but does not mention any particular methods of parameter efficient tuning and thus may not be super helpful. This could still be useful to discuss general advantages of PEFT.
[1] Rating: 5 Explanation: This paragraph introduces Prefix Tuning, one of the most representative methods in parameter efficient tuning and includes their core empirical results.
[2] Rating: 3 Explanation: While this paragraph provides a taxonomy of parameter efficient tuning and analysis, it does not provide any details of individual methods. Moreover, this paper's main focus is PEFT for vision models, while the original question asks about parameter efficient tuning for large language models.
[3] Rating: 4 Explanation: This paragraph briefly introduces multiple parameter efficient tuning methods such as adapter tuning, prefix tuning and prompt tuning. While they do not directly discuss their advantages or disadvantages or more detail about prefix or prompt tuning, still this paragraph gives a useful overview of this area.
[4] Rating: 1 Explanation: This paragraph introduces a new parameter efficient tuning method to connect a vision encoder and language encoder to make their representations aligned. The question asks about representative approaches of parameter efficient tuning for large language models, and this paragraph topic is substantially different from the question.[Response_End]\n
"""

prompts_w_references = ("Provide a detailed, informative answer to the following research-related question. Your answer should be more than one paragraph, offering a comprehensive overview. "
                       "Base your answer on multiple pieces of evidence and references, rather than relying on a single reference for a short response. Aim to give a holistic view of the topic. "
                       "Ensure the answer is well-structured, coherent and informative so that real-world scientists can gain a clear understanding of the subject. Rather than simply summarizing multiple papers one by one, try to organize your answers based on similarities and differences between papers. " 
                       "Make sure to add citations to all citation-worthy statements using the provided references (References), by indicating the citation numbers of the corresponding passages. "
                       "More specifically, add the citation number at the end of each relevant sentence e.g., 'This work shows the effectiveness of problem X [1].' when the passage [1] in References provides full support for the statement. "
                       "You do not need to add the author names, title or publication year as in the ordinal paper writing, and just mention the citation numbers with your generation. "
                       "Not all references may be relevant, so only cite those that directly support the statement. "
                       "You only need to indicate the reference number, and you do not need to add Reference list by yourself. "
                       "If multiple references support a statement, cite them together (e.g., [1][2]). Yet, for each citation-worthy statement, you only need to add at least one citation, so if multiple eviences support the statement, just add the most relevant citation to the sentence. "
                       "Your answer should be marked as [Response_Start] and [Response_End].\n"
                       "Here's an example:\n##\n"
                       "References: \n{example_passages}"
                       "\nQuestion: {example_question}"
                       "\n[Response_Start]{example_answer}[Response_End]\nNow, please answer this question\n##\n")
generation_demonstration_prompts = prompts_w_references.format_map({"example_passages": example_passages_rag, "example_question": example_question_rag, "example_answer": example_answer_rag})
generation_instance_prompts_w_references = generation_demonstration_prompts + "References:\n {context}\nQuestion: {input}\n"

generation_instance_prompts_w_references_zero_shot = ("Provide a detailed, informative answer to the following research-related question. Your answer should be more than one paragraph, offering a comprehensive overview."
                       "Base your answer on multiple pieces of evidence and references, rather than relying on a single reference for a short response. Aim to give a holistic view of the topic."
                       "Ensure the answer is well-structured, coherent and informative so that real-world scientists can gain a clear understanding of the subject. Rather than simply summarizing multiple papers one by one, try to organize your answers based on similarities and differences between papers." 
                       "Make sure to add citations to all citation-worthy statements using the provided references (References). More specifically, add the citation number at the end of each relevant sentence e.g., 'This work shows the effectiveness of problem X [1].' when the passage [1] in References provides full support for the statement."
                       "Not all references may be relevant, so only cite those that directly support the statement."
                       "If multiple references support a statement, cite them together (e.g., [1][2]). Yet, for each citation-worthy statement, you only need to add at least one citation, so if multiple eviences support the statement, just add the most relevant citation to the sentence."
                       "References: \n{context}"
                       "\nQuestion: {input}")

# Ranking
prompts_reranking = """
Evaluate the relevance of passages from scientific papers to aid in crafting informative responses to a given question. Each passage should be assessed for its potential contribution to understanding the question, including clear definitions, method advantages, method comparisons, and concrete experimental results. Each passage should be rated on a scale from 1 to 5, with explanations provided for each rating:\n
1 (Completely Irrelevant): The passage is entirely unrelated to the question and does not offer any relevant information. Your answer should be marked as [Response_Start] and [Response_End].\n
2 (Somewhat Related): The passage is tangentially related to the question but does not provide any substantial information that could be incorporated into an answer.\n
3 (Partially Relevant): Although the passage does not directly address the question, it offers high-level information that could enhance non-essential parts of the answer, such as opening remarks or supplemental experimental results.\n
4 (Relevant): The passage provides useful information that could be integrated into essential parts of the answer, contributing significantly to understanding the question.\n
5 (Highly Relevant): The passage contains important information, and several sentences could be directly cited and used in the response to provide crucial insights into the question.\n
"""

ranking_example_instance_prompt = "{prompts_reranking}\n##\nQuestion: {example_question}\nReferences\n{example_paragraph}\n{example_rating}".format_map({"prompts_reranking": prompts_reranking, "example_question": example_question_peft, "example_paragraph": example_passages_peft, "example_rating": example_rating_peft })
ranking_instance_prompt = ranking_example_instance_prompt + "\n##\nQuestion: {question}\nReferences\n{passages}\n"

# Updated on June 15
prompts_reranking_summarization = """
Evaluate the relevance of passages from scientific papers to aid in crafting informative related work section to a given abstract. Each passage should be assessed for its potential contribution to understanding the original abstract, including clear definitions, method advantages, method comparisons, and concrete experimental results. Each passage should be rated on a scale from 1 to 5, with explanations provided for each rating:\n
1 (Completely Irrelevant): The passage is entirely unrelated to the ideal related work section given the abstract, and does not offer any relevant information. Your answer should be marked as [Response_Start] and [Response_End].\n
2 (Somewhat Related): The passage is tangentially related to the topic discussed in the abstract and may help users to write related work, but does not provide any substantial information that could be incorporated into the related work.\n
3 (Partially Relevant): Although the passage may not provide crucial information to write a related work section, it offers some useful information that could enhance non-essential parts of the related work, such as opening remarks or supplemental experimental results, as well as additional exampels to strengthen main arguements.\n
4 (Relevant): The passage provides useful information that could be integrated into essential parts of the related work, contributing significantly to understand and contextualized the abstract.\n
5 (Highly Relevant): The passage contains important information, and several sentences could be directly cited and used in the related work to provide crucial insights into the related work users try to write.\n
"""
# TODO: add demonstrations
example_question_summarization = "We present QuAC, a dataset for Question Answering in Context that contains 14K information-seeking QA dialogs (100K questions in total). The dialogs involve two crowd workers: (1) a student who poses a sequence of freeform questions to learn as much as possible about a hidden Wikipedia text, and (2) a teacher who answers the questions by providing short excerpts from the text. QuAC introduces challenges not found in existing machine comprehension datasets: its questions are often more open-ended, unanswerable, or only meaningful within the dialog context, as we show in a detailed qualitative evaluation. We also report results for a number of reference models, including a recently state-of-the-art reading comprehension architecture extended to model dialog context. Our best model underperforms humans by 20 F1, suggesting that there is significant room for future work on this data. "

example_passages_summarization = """
[0] Title: CoQA: A Conversational Question Answering Challenge Text: Humans gather information by engaging in conversations involving a series of interconnected questions and answers. For machines to assist in information gathering, it is therefore essential to enable them to answer conversational questions. We introduce CoQA, a novel dataset for building Conversational Question Answering systems. Our dataset contains 127k questions with answers, obtained from 8k conversations about text passages from seven diverse domains. The questions are conversational, and the answers are free-form text with their corresponding evidence highlighted in the passage. We analyze CoQA in depth and show that conversational questions have challenging phenomena not present in existing reading comprehension datasets, e.g., coreference and pragmatic reasoning. We evaluate strong conversational and reading comprehension models on CoQA. The best system obtains an F1 score of 65.4\%, which is 23.4 points behind human performance (88.8\%), indicating there is ample room for improvement. \n
[1] Title: SQuAD: 100,000+ Questions for Machine Comprehension of Text Text: We present the Stanford Question Answering Dataset (SQuAD), a new reading comprehension dataset consisting of 100,000+ questions posed by crowdworkers on a set of Wikipedia articles, where the answer to each question is a segment of text from the corresponding reading passage. We analyze the dataset to understand the types of reasoning required to answer the questions, leaning heavily on dependency and constituency trees. We build a strong logistic regression model, which achieves an F1 score of 51.0\%, a significant improvement over a simple baseline (20\%). However, human performance (86.8\%) is much higher, indicating that the dataset presents a good challenge problem for future research.\n
[2] Title: Interpretation of natural language rules in conversational machine reading Text: Most work in machine reading focuses on question answering problems where the answer is directly expressed in the text to read. However, many real-world question answering problems require the reading of text not because it contains the literal answer, but because it contains a recipe to derive an answer together with the reader's background knowledge. One example is the task of interpreting regulations to answer "Can I...?" or "Do I have to...?" questions such as "I am working in Canada. Do I have to carry on paying UK National Insurance?" after reading a UK government website about this topic. This task requires both the interpretation of rules and the application of background knowledge. It is further complicated due to the fact that, in practice, most questions are underspecified, and a human assistant will regularly have to ask clarification questions such as How long have you been working abroad? when the answer cannot be directly derived from the question and text. In this paper, we formalise this task and develop a crowd-sourcing strategy to collect 32k task instances based on real-world rules and crowd-generated questions and scenarios. We analyse the challenges of this task and assess its difficulty by evaluating the performance of rule-based and machine-learning baselines. We observe promising results when no background knowledge is necessary, and substantial room for improvement whenever background knowledge is needed.\n
[3] Title: Passage Re-ranking with BERT Text: Recently, neural models pretrained on a language modeling task, such as ELMo (Peters et al., 2017), OpenAI GPT (Radford et al., 2018), and BERT (Devlin et al., 2018), have achieved impressive results on various natural language processing tasks such as question-answering and natural language inference. In this paper, we describe a simple re-implementation of BERT for query-based passage re-ranking. Our system is the state of the art on the TREC-CAR dataset and the top entry in the leaderboard of the MS MARCO passage retrieval task, outperforming the previous state of the art by 27\% (relative) in MRR@10. \n
[4] Title: Bidirectional Attention Flow for Machine Comprehension Text: Machine comprehension (MC), answering a query about a given context paragraph, requires modeling complex interactions between the context and the query. Recently, attention mechanisms have been successfully extended to MC. Typically these methods use attention to focus on a small portion of the context and summarize it with a fixed-size vector, couple attentions temporally, and/or often form a uni-directional attention. In this paper we introduce the Bi-Directional Attention Flow (BIDAF) network, a multi-stage hierarchical process that represents the context at different levels of granularity and uses bi-directional attention flow mechanism to obtain a query-aware context representation without early summarization. Our experimental evaluations show that our model achieves the state-of-the-art results in Stanford Question Answering Dataset (SQuAD) and CNN/DailyMail cloze test.\n
"""

example_rating_summarization = """
[Response_Start][0] Rating: 5 Explanation: This paper seems to introduce a dataset that sounds really similar to the proposed dataset and the authors definitely need to discuss how their proposed dataaset is related to this paper. Therefore, the rating is 5.\n
[1] Rating: 4 Explanation: This paper introduces a large-scale machine reading comprehension dataset. Although this dataset is not a conversational QA dataset proposed by the original abstract and the set up may be more simple than conversational QA, still this dataset could be useful to cite when the authors discuss the history and different datasets in relavent areas.\n
[2] Rating: 5 Explanation: This paper presents a task of generating and answering yes/no questions for rule focused text (such as traffic laws) by interacting with a user through dialog. This paper also considers a conversational QA situation and is highly relavant to the abstract.\n
[3] Rating: 2 Explanation: This paper introduces a new method for passage ranking to enhance information retrieval, using a fine-tuned pre-trained encoder model, namely BERT. While this method may or may not be used in the paper of the subject abstract, this method is mainly proposed for IR and is less relevant to the proposed conversational QA setup. Therefore, this paper may not provide useful information to be included in the related work section.\n 
[4] Rating: 3 Explanation: This paper proposes a new method that performs competitively on a machine reading comprehension dataset, SQuAD. Although this paper could be cited to discuss how a more simple machine reading comprehension task has evolved in terms of datasets and methodologies, given that the main focus of the provided abstract is on conversational QA, the method paper on MRC may not be really crucial.[Response_End]\n
"""

example_answer_summarization = """
Our work builds on span based reading comprehension [1] while also incorporating innovations such as curating questions independently of supporting text to reduce trivial lexical overlap. 
Concurrent to our work, [2] proposed a task of generating and answering yes/no questions for rule focused text (such as traffic laws) by interacting with a user through dialog. 
Also concurrently, [0] propose conversational question answering (CoQA) from text but allow both students and questioners to see the evidence. 
"""


# TODO: add examples
example_passages_single_paper = """
[0] We hypothesize that factual knowledge frequently discussed on the web is easily memorized by LMs, while the knowledge that is less discussed may not be well captured and thus they require retrieving external non-parametric memories. We evaluate ten large LMs of three families (i.e., GPT-Neo, OPT, and GPT-3) with varying scales on the open-domain question answering (QA) task in a zero- or few-shot prompting manner. 
[1] We construct a new dataset, PopQA, consisting of 14k questions to cover factual information in the long tail that might have been missed in popular QA datasets Kwiatkowski et al. (2019). We use Wikipedia page views as a measure of popularity and convert knowledge triples from Wikidata, with diverse levels of popularity, into natural language questions, anchored to the original entities and relationship types. We also use EntityQuestions Sciavolino et al. (2021), an open-domain QA dataset with a long-tail distribution.
[2] Figure 4 (bottom) shows that there is a positive correlation between subject entity popularity and models' accuracy for almost all relationship types. This supports our hypothesis that subject entity popularity can be a reliable indicator of LMs' factual knowledge memorization. In general, the correlations between subject entity popularity and accuracy are stronger for larger LMs; GPT-3 003 shows the highest positive correlation (roughly 0.4) while GPT-Neo-1.3B shows relatively weak positive correlations (approximately 0.1).
[3] As seen in the left column of Figure 4, there are clear overall performance improvements with scale on the PopQA dataset. However, Figure 5 shows that on both PopQA and EntityQuestions, most of scaling's positive effect on parametric knowledge comes from questions with high popularity. Specifically, for the questions about the entities whose log 10 (subject popularity) is larger than 4, there is an improvement in accuracy as model size increases (red and yellow lines), while performance on questions with lower popularity remains relatively constant (blue and green lines). For the 4,000 least popular questions, GPT-Neo 6B, 20B, and GPT-3 davinci-003 have 15\%, 16\%, and 19\% accuracy, respectively.
[4] Our analysis indicates that even the current state-of-the-art LMs struggle with less popular subjects or certain relationship types, and increasing the model size does not lead to further performance improvements. In light of this, we extend our analysis to non-parametric sources of knowledge, as outlined in Section 2. Specifically, we investigate the effectiveness of retrieval-augmented LMs Borgeaud et al. (2022); Lewis et al. (2020), which leverage non-parametric memories (i.e., retrieved text) to improve performance.
[5] Figure 7 shows that augmenting LMs with non-parametric memories significantly outperforms unassisted vanilla LMs. A much smaller LM (e.g., GPT-Neo 2.7B) augmented by the Contriever retrieval results outperforms vanilla GPT-3. Large LMs such as GPT-3 also enjoy the benefits of non-parametric memories. Contriever gives 7\% accuracy gains on top of GPT-3 davinci-003. GenRead shows little-to-no performance improvement over vanilla parametric knowledge for smaller models, while the technique shows sizeable gains for GPT-3, especially davinci-003. In addition to its limited effectiveness with smaller LMs, GenRead has potentially prohibitive inference time costs, with GPT-NeoX 20B taking 70 seconds per query.
"""
example_question_single_paper = "What is the authors' conclusion about the effectiveness of language model scaling on long-tail factual knowledge memorization?"
example_answer_single_paper = "The authors found that on their newly constructed dataset, Pop QA [1], performance on questions with lower popularity (long tail facts) remains relatively constant [3]. The authors concluded model scaling may not help long-tail factual memorization [4]."

example_answer_single_paper_no_context = "The authors found that on their newly constructed dataset, Pop QA, performance on questions with lower popularity (long tail facts) remains relatively constant. The authors concluded model scaling may not help long-tail factual memorization."



ranking_example_instance_prompt_summarization = "{prompts_reranking}\n##\nAbstract: {example_question}\nPassages:\n{example_paragraph}\n{example_rating}".format_map({"prompts_reranking": prompts_reranking_summarization, "example_question": example_question_summarization, "example_paragraph": example_passages_summarization, "example_rating": example_rating_summarization})
ranking_instance_prompt_summarization = ranking_example_instance_prompt_summarization + "\n##\nAbstract: {question}\nPassages:\n{passages}\n"

# Feedback
instruction_feedback = """
Given an answer to a scientific question based on the most recent scientific literature, make a list of feedback. Prioritize the feedback by listing the most critical improvements first. Regarding the content improvements, it is often helpful to ask for more results on or applications to different tasks, elaborate on details of crucial methods, or suggest including other popular methods.
Stylistic improvements can include better organizations or writing enhancements. For each suggested improvement requiring additional information from the literature that is not discussed in passages, formulate a question to guide the search for missing details.
If the feedback primarily concerns stylistic or organizational changes, omit the need for an additional question. Your answer should be marked as [Response_Start] and [Response_End].
Each feedback should be preceded by 'Feedback: ', and additional question should be preceded by 'Question: '. Your question will be used to search additional context, so they should be self-containing and are understandable without additional context. 
Here's an example.\n
##\n
Question: {example_question}\n
Answer: {example_answer}\n
[Response_Start]{example_feedback}[Response_End]
Now, please generate feedback for this question.
##\n
"""
instruction_feedback_prompt = instruction_feedback.format_map({"example_question": example_question_rag, \
    "example_answer":example_answer_rag_incorrect, \
    "example_paragraphs": example_passages_rag, \
    "example_feedback": example_feedback})
feedback_example_instance_prompt = instruction_feedback_prompt +  "Question: {question}\nAnswer:\n{answer}\n"


editing_feedback = """
We provide a question related to recent scientific literature, an answer from a strong language model, and feedback on the answer.
Please incorporate the feedback to improve the answer. Only modify the parts that require enhancement as noted in the feedback, keeping the other sentences unchanged.
Do not omit any crucial information from the original answer unless the feedback specifies that certain sentences are incorrect and should be removed.
If you add new paragraphs or discussions, ensure that you are not introducing repetitive content or duplicating ideas already included in the original response.
Use existing references presented under References to support the new discussions, referring to their citation numbers. 
Do not remove new lines or paragraphs in the original answer, unless the feedback specifies that certain sentences are incorrect and should be removed, or the paragraph organizations should be changed. 
Your answer should be marked as [Response_Start] and [Response_End].\n
References:
[0] Title: Attributed Question Answering: Evaluation and Modeling for Attributed Large Language Models Text: Roberts et al. (2020) shows that T5 (Raffel et al., 2020) can perform a new task formulation, closedbook QA. Concretely, T5 can produce answers to questions without access to any corpus at inference time, instead producing answers based on its model parameters, tuned to remember information digested in pretraining.\n
[1] Title: Reliable, Adaptable, and Attributable Language Models with Retrieval Text: Unlike parametric LMs—which use large-scale text data only during training—retrieval-augmented LMs leverage an external large-scale collection of documents (datastore) at inference by selecting relevant documents from the datastore (Asai et al., 2023a). Retrieval-augmented LMs can W1: largely reduce factual errors (Mallen et al., 2023), W2: provide better attributions (Gao et al., 2023a), W3: enabling flexible opt-in and out of sequences (Min et al., 2024).
[2] Title: Atlas: Few-shot Learning with Retrieval Augmented Language Models Text: In this work we present Atlas, a carefully designed and pre-trained retrieval augmented language model able to learn knowledge intensive tasks with very few training examples. We perform evaluations on a wide range of tasks, including MMLU, KILT and NaturalQuestions, and study the impact of the content of the document index, showing that it can easily be updated. Notably, Atlas reaches over 42% accuracy on Natural Questions using only 64 examples, outperforming a 540B parameters model by 3% despite having 50x fewer parameters.
[3] Title: Language Models are Few-Shot Learners Text: Similarly, GPT-3 achieves 64.3% accuracy on TriviaQA in the zero-shot setting, 68.0% in the one-shot setting, and 71.2% in the few-shot setting, the last of which is state-of-the-art relative to fine-tuned models operating in the same closed-book setting.
Question: What are the advantages of retrieval-augmented LMs?
Answer: Retrieval-augmented LMs have been effective in various use cases, including reducing hallucinations [0] and are often more parameter-efficient than non retrieval-augmented LMs [2].
Feedback: 
The answer provides only list advantages without providing any concrete examples. Please provide more examples of how retrieval-augmented LMs have been used in practice.
Edited Answer:
[Response_Start]Retrieval-augmented LMs have been effective in various use cases, including reducing hallucinations [0] and are often more parameter-efficient than non retrieval-augmented LMs [2]. For instance, Atlas [2] achieves 42% accuracy on Natural Questions using only 64 examples, outperforming a 540B parameters model by 3\% despite having 50x fewer parameters.[Response_End]
"""

editing_instance_prompt = \
    editing_feedback +  "##\nReferences\n{passages}\nQuestion: {question}\nAnswer:\n{answer}\nFeedback:\n{feedback}\nEdited Answer:\n"

posthoc_attributions = """
We give you a statement extracted from an answer to a question related to the most recent scientific literature, and a set of evidence passages.
If the statement is fully supported by any of the listed passages in References, insert the citation numbers to the statement.
If none of the passages support the statement, do not insert any citation, and leave the original sentence as is.
If multiple passages provide sufficient support for the statement, you only need to insert one citation, rather than inserting all of them. Your answer should be marked as [Response_Start] and [Response_End].'\n
Here's an example:\n
Statement: On the other hand, non-parametric knowledge is retrieved from an external source, such as a large-scale collection of documents, during inference.
References:
[0] Title: Attributed Question Answering: Evaluation and Modeling for Attributed Large Language Models Text: Roberts et al. (2020) shows that T5 (Raffel et al., 2020) can perform a new task formulation, closedbook QA. Concretely, T5 can produce answers to questions without access to any corpus at inference time, instead producing answers based on its model parameters, tuned to remember information digested in pretraining.\n
[1] Title: Reliable, Adaptable, and Attributable Language Models with Retrieval Text: Unlike parametric LMs—which use large-scale text data only during training—retrieval-augmented LMs leverage an external large-scale collection of documents (datastore) at inference by selecting relevant documents from the datastore (Asai et al., 2023a). Retrieval-augmented LMs can W1: largely reduce factual errors (Mallen et al., 2023), W2: provide better attributions (Gao et al., 2023a), W3: enabling flexible opt-in and out of sequences (Min et al., 2024).
[2] Title: Atlas: Few-shot Learning with Retrieval Augmented Language Models Text: In this work we present Atlas, a carefully designed and pre-trained retrieval augmented language model able to learn knowledge intensive tasks with very few training examples. We perform evaluations on a wide range of tasks, including MMLU, KILT and NaturalQuestions, and study the impact of the content of the document index, showing that it can easily be updated. Notably, Atlas reaches over 42\% accuracy on Natural Questions using only 64 examples, outperforming a 540B parameters model by 3\% despite having 50x fewer parameters.
[3] Title: Language Models are Few-Shot Learners Text: Similarly, GPT-3 achieves 64.3\% accuracy on TriviaQA in the zero-shot setting, 68.0\% in the one-shot setting, and 71.2\% in the few-shot setting, the last of which is state-of-the-art relative to fine-tuned models operating in the same closed-book setting.
[4] Title: When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories Text:  On both datasets, LMs' memorization (RQ1) is often limited to the popular factual knowledge and even GPT-3 davinci-003 fails to answer the majority of the long-tail questions. Moreover, on such questions, scaling up models does not significantly improve the performance. This also suggests that we can predict if LMs memorize certain knowledge based on the information presented in the input question only. We next investigate whether a semi-parametric approach that augments LMs with retrieved evidence can mitigate the low performance on questions about less popular entities (RQ2). Nonparametric memories largely improve performance on long-tail distributions across models.
[5] Title: Democratizing Large Language Models via Personalized Parameter-Efficient Fine-tuning Text: Personalization in large language models (LLMs) is increasingly important, aiming to align LLM's interactions, content, and recommendations with individual user preferences. Recent advances in LLM personalization have spotlighted effective prompt design, by enriching user queries with non-parametric knowledge through behavior history retrieval and textual profiles. However, these approaches were limited due to a lack of model ownership, resulting in constrained customization and privacy issues. Moreover, they often failed to accurately capture user behavior patterns, especially in cases where user data were complex and dynamic. To address these shortcomings, we introduce One PEFT Per User (OPPU), which employs personalized parameter-efficient fine-tuning (PEFT) modules, to store user-specific behavior patterns and preferences.
[6] Title: RECOMP: Improving Retrieval-Augmented LMs with Context Compression and Selective Augmentation Text:  Retrieval-augmented language models (RALMs) (Khandelwal et al., 2019; Izacard et al., 2022; Lewis et al., 2020; Borgeaud et al., 2022) have shown impressive performance on knowledge-intensive tasks (Kwiatkowski et al., 2019; Petroni et al., 2021). Simply prepending retrieved documents to the input without updating the language models (LMs) (Shi et al., 2023b; Ram et al., 2023; Si et al., 2022) allows retrieval augmentation even for black-box LMs, but such approach comes with limitations. First, it increases computational costs as LMs now encode substantially more tokens. Second, even if we manage to adapt LMs to efficiently incorporate longer context (Beltagy et al., 2020; Zaheer et al., 2020), these models struggle to use all information in the context, frequently missing information placed in the middle (Liu et al., 2023). Third, prepending a large number of documents in-context can further confuse LMs with irrelevant information, degrading model performances (Mallen et al., 2022; Shi et al., 2023a).
[Response_Start]On the other hand, non-parametric knowledge is retrieved from an external source, such as a large-scale collection of documents, during inference [1].[Response_End]\n
Now, please insert citations to the following sentence. ##\n
Statement: {statement}
References:\n{passages}\n
"""

posthoc_attributions_paragraph_all = """
We give you a short paragraph extracted from an answer to a question related to the most recent scientific literature, and a set of evidence passages.
Find all of the citation-worthy statements without any citations, and insert citation numbers to the statements that are fully supported by any of the provided citations in listed as References. 
If none of the passages support the statement, do not insert any citation, and leave the original sentence as is, but do your best to insert citation. 
If multiple passages provide sufficient support for the statement, you only need to insert one citation, rather than inserting all of them. Your answer should be marked as [Response_Start] and [Response_End].'\n
Here's an example:\n
Statement: Language models store rich knowledge in their parameters during pre-training, resulting in their strong performance on many knowledge-intensive tasks. However, such parametric knowledge based generations are often hard to attribute. Models can also struggle in long-tail knowledge. On the other hand, non-parametric knowledge is retrieved from an external source, such as a large-scale collection of documents, during inference. 
References:
[0] Title: Attributed Question Answering: Evaluation and Modeling for Attributed Large Language Models Text: Roberts et al. (2020) shows that T5 (Raffel et al., 2020) can perform a new task formulation, closedbook QA. Concretely, T5 can produce answers to questions without access to any corpus at inference time, instead producing answers based on its model parameters, tuned to remember information digested in pretraining.\n
[1] Title: Reliable, Adaptable, and Attributable Language Models with Retrieval Text: Unlike parametric LMs—which use large-scale text data only during training; retrieval-augmented LMs leverage an external large-scale collection of documents (datastore) at inference by selecting relevant documents from the datastore (Asai et al., 2023a). Retrieval-augmented LMs can W1: largely reduce factual errors (Mallen et al., 2023), W2: provide better attributions (Gao et al., 2023a), W3: enabling flexible opt-in and out of sequences (Min et al., 2024).
[2] Title: Atlas: Few-shot Learning with Retrieval Augmented Language Models Text: In this work we present Atlas, a carefully designed and pre-trained retrieval augmented language model able to learn knowledge intensive tasks with very few training examples. We perform evaluations on a wide range of tasks, including MMLU, KILT and NaturalQuestions, and study the impact of the content of the document index, showing that it can easily be updated. Notably, Atlas reaches over 42\% accuracy on Natural Questions using only 64 examples, outperforming a 540B parameters model by 3\% despite having 50x fewer parameters.
[3] Title: Language Models are Few-Shot Learners Text: Similarly, GPT-3 achieves 64.3\% accuracy on TriviaQA in the zero-shot setting, 68.0\% in the one-shot setting, and 71.2\% in the few-shot setting, the last of which is state-of-the-art relative to fine-tuned models operating in the same closed-book setting.
[4] Title: When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories Text:  On both datasets, LMs' memorization (RQ1) is often limited to the popular factual knowledge and even GPT-3 davinci-003 fails to answer the majority of the long-tail questions. Moreover, on such questions, scaling up models does not significantly improve the performance. This also suggests that we can predict if LMs memorize certain knowledge based on the information presented in the input question only. We next investigate whether a semi-parametric approach that augments LMs with retrieved evidence can mitigate the low performance on questions about less popular entities (RQ2). Nonparametric memories largely improve performance on long-tail distributions across models.
[5] Title: Democratizing Large Language Models via Personalized Parameter-Efficient Fine-tuning Text: Personalization in large language models (LLMs) is increasingly important, aiming to align LLM's interactions, content, and recommendations with individual user preferences. Recent advances in LLM personalization have spotlighted effective prompt design, by enriching user queries with non-parametric knowledge through behavior history retrieval and textual profiles. However, these approaches were limited due to a lack of model ownership, resulting in constrained customization and privacy issues. Moreover, they often failed to accurately capture user behavior patterns, especially in cases where user data were complex and dynamic. To address these shortcomings, we introduce One PEFT Per User (OPPU), which employs personalized parameter-efficient fine-tuning (PEFT) modules, to store user-specific behavior patterns and preferences.
[6] Title: RECOMP: Improving Retrieval-Augmented LMs with Context Compression and Selective Augmentation Text:  Retrieval-augmented language models (RALMs) (Khandelwal et al., 2019; Izacard et al., 2022; Lewis et al., 2020; Borgeaud et al., 2022) have shown impressive performance on knowledge-intensive tasks (Kwiatkowski et al., 2019; Petroni et al., 2021). Simply prepending retrieved documents to the input without updating the language models (LMs) (Shi et al., 2023b; Ram et al., 2023; Si et al., 2022) allows retrieval augmentation even for black-box LMs, but such approach comes with limitations. First, it increases computational costs as LMs now encode substantially more tokens. Second, even if we manage to adapt LMs to efficiently incorporate longer context (Beltagy et al., 2020; Zaheer et al., 2020), these models struggle to use all information in the context, frequently missing information placed in the middle (Liu et al., 2023). Third, prepending a large number of documents in-context can further confuse LMs with irrelevant information, degrading model performances (Mallen et al., 2022; Shi et al., 2023a).
[Response_Start]Language models store rich knowledge in their parameters during pre-training, resulting in their strong performance on many knowledge-intensive tasks [3]. However, such parametric knowledge based generations are often hard to attribute [0]. Models can also struggle in long-tail knowledge [4]. On the other hand, non-parametric knowledge is retrieved from an external source, such as a large-scale collection of documents, during inference [2].[Response_End]\n
Now, please insert citations to the following sentece. ##\n
Statement: {statement}
References:\n{passages}\n
"""

posthoc_attributions_paragraph = """
We give you a short paragraph extracted from an answer to a question related to the most recent scientific literature, and a set of evidence passages.
Find all of the citation-worthy statements without any citations, and insert citation numbers to the statements that are fully supported by any of the provided citations in listed as References. 
If none of the passages support the statement, do not insert any citation number, and leave the original sentence as is. 
You onlyneed to add a citation number if applicable, and do not need to modify the original text. Do not directly insert text from the relevant evidence.
If multilpe passages provide sufficient support for the statement, you only need to insert one citation, rather than inserting all of them. Your answer should be marked as [Response_Start] and [Response_End].'\n
Here's an example:\n
Statement: Language models store rich knowledge in their parameters during pre-training, resulting in their strong performance on many knowledge-intensive tasks. However, such parametric knowledge based generations are often hard to attribute. Models can also struggle in long-tail knowledge. On the other hand, non-parametric knowledge is retrieved from an external source, such as a large-scale collection of documents, during inference. 
References:
[0] Title: Attributed Question Answering: Evaluation and Modeling for Attributed Large Language Models Text: Roberts et al. (2020) shows that T5 (Raffel et al., 2020) can perform a new task formulation, closedbook QA. Concretely, T5 can produce answers to questions without access to any corpus at inference time, instead producing answers based on its model parameters, tuned to remember information digested in pretraining.\n
[1] Title: Reliable, Adaptable, and Attributable Language Models with Retrieval Text: Unlike parametric LMs—which use large-scale text data only during training; retrieval-augmented LMs leverage an external large-scale collection of documents (datastore) at inference by selecting relevant documents from the datastore (Asai et al., 2023a). Retrieval-augmented LMs can W1: largely reduce factual errors (Mallen et al., 2023), W2: provide better attributions (Gao et al., 2023a), W3: enabling flexible opt-in and out of sequences (Min et al., 2024).
[2] Title: Atlas: Few-shot Learning with Retrieval Augmented Language Models Text: In this work we present Atlas, a carefully designed and pre-trained retrieval augmented language model able to learn knowledge intensive tasks with very few training examples. We perform evaluations on a wide range of tasks, including MMLU, KILT and NaturalQuestions, and study the impact of the content of the document index, showing that it can easily be updated. Notably, Atlas reaches over 42\% accuracy on Natural Questions using only 64 examples, outperforming a 540B parameters model by 3\% despite having 50x fewer parameters.
[3] Title: Language Models are Few-Shot Learners Text: Similarly, GPT-3 achieves 64.3\% accuracy on TriviaQA in the zero-shot setting, 68.0\% in the one-shot setting, and 71.2\% in the few-shot setting, the last of which is state-of-the-art relative to fine-tuned models operating in the same closed-book setting.
[4] Title: When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories Text:  On both datasets, LMs' memorization (RQ1) is often limited to the popular factual knowledge and even GPT-3 davinci-003 fails to answer the majority of the long-tail questions. Moreover, on such questions, scaling up models does not significantly improve the performance. This also suggests that we can predict if LMs memorize certain knowledge based on the information presented in the input question only. We next investigate whether a semi-parametric approach that augments LMs with retrieved evidence can mitigate the low performance on questions about less popular entities (RQ2). Nonparametric memories largely improve performance on long-tail distributions across models.
[5] Title: Democratizing Large Language Models via Personalized Parameter-Efficient Fine-tuning Text: Personalization in large language models (LLMs) is increasingly important, aiming to align LLM's interactions, content, and recommendations with individual user preferences. Recent advances in LLM personalization have spotlighted effective prompt design, by enriching user queries with non-parametric knowledge through behavior history retrieval and textual profiles. However, these approaches were limited due to a lack of model ownership, resulting in constrained customization and privacy issues. Moreover, they often failed to accurately capture user behavior patterns, especially in cases where user data were complex and dynamic. To address these shortcomings, we introduce One PEFT Per User (OPPU), which employs personalized parameter-efficient fine-tuning (PEFT) modules, to store user-specific behavior patterns and preferences.
[6] Title: RECOMP: Improving Retrieval-Augmented LMs with Context Compression and Selective Augmentation Text:  Retrieval-augmented language models (RALMs) (Khandelwal et al., 2019; Izacard et al., 2022; Lewis et al., 2020; Borgeaud et al., 2022) have shown impressive performance on knowledge-intensive tasks (Kwiatkowski et al., 2019; Petroni et al., 2021). Simply prepending retrieved documents to the input without updating the language models (LMs) (Shi et al., 2023b; Ram et al., 2023; Si et al., 2022) allows retrieval augmentation even for black-box LMs, but such approach comes with limitations. First, it increases computational costs as LMs now encode substantially more tokens. Second, even if we manage to adapt LMs to efficiently incorporate longer context (Beltagy et al., 2020; Zaheer et al., 2020), these models struggle to use all information in the context, frequently missing information placed in the middle (Liu et al., 2023). Third, prepending a large number of documents in-context can further confuse LMs with irrelevant information, degrading model performances (Mallen et al., 2022; Shi et al., 2023a).
[Response_Start]Language models store rich knowledge in their parameters during pre-training, resulting in their strong performance on many knowledge-intensive tasks [3]. However, such parametric knowledge based generations are often hard to attribute [0]. Models can also struggle in long-tail knowledge [4]. On the other hand, non-parametric knowledge is retrieved from an external source, such as a large-scale collection of documents, during inference [2].[Response_End]\n
Now, please insert citations to the following sentece. ##\n
Statement: {statement}
References:\n{passages}\n
"""

posthoc_attribution_with_citations = """
We give you a statement extracted from an answer to a question related to the most recent scientific literature, and a cited evidence passage.
Evaluate if the statement is fully supported by the citation, and answer with 'Yes' (the statement is fully supported by the citation) or 'No' (otherwise), Your rating and explanations should be indicated as 'Rating:' and 'Explanation:', after Your answer should be marked, and your response should be marked as [Response_Start] and [Response_End].
After the rating, please explain why it is / is not fully supported.\n
##\n
Statement: Specifically, retrieval-augmented LMs can make inference much more inefficient due to increased context length.
References:\n
Title: RECOMP: Improving Retrieval-Augmented LMs with Context Compression and Selective Augmentation Text:  Retrieval-augmented language models (RALMs) (Khandelwal et al., 2019; Izacard et al., 2022; Lewis et al., 2020; Borgeaud et al., 2022) have shown impressive performance on knowledge-intensive tasks (Kwiatkowski et al., 2019; Petroni et al., 2021). Simply prepending retrieved documents to the input without updating the language models (LMs) (Shi et al., 2023b; Ram et al., 2023; Si et al., 2022) allows retrieval augmentation even for black-box LMs, but such approach comes with limitations. First, it increases computational costs as LMs now encode substantially more tokens. Second, even if we manage to adapt LMs to efficiently incorporate longer context (Beltagy et al., 2020; Zaheer et al., 2020), these models struggle to use all information in the context, frequently missing information placed in the middle (Liu et al., 2023). Third, prepending a large number of documents in-context can further confuse LMs with irrelevant information, degrading model performances (Mallen et al., 2022; Shi et al., 2023a).
[Response_Start]Rating: Yes\n
Explanation: The cited passage explicitly mentions that retrieval-augmented LMs will introduce additional inference time latency. [Response_End]\n
##\n
Statement: Compared to other parameter efficient tuning method, adapters have shown to be more stable.
References:\n
Title: Towards a Unified View of Parameter-Efficient Transfer Learning Text: To mitigate this issue, a few lightweight alternatives have been proposed to update only a small number of extra parameters while keeping most pretrained parameters frozen. For example, adapter tuning (Houlsby et al., 2019) inserts small neural modules called adapters to each layer of the pretrained network and only the adapters are trained at fine-tuning time. Inspired by the success of prompting methods that control PLMs through textual prompts (Brown et al., 2020; Liu et al., 2021a), prefix tuning (Li & Liang, 2021) and prompt tuning (Lester et al., 2021) prepend an additional l tunable prefix tokens to the input or hidden layers and only train these soft prompts when fine-tuning on downstream tasks.
[Response_Start]Rating: No\n
Explanation: The provided citation discusses several parameter efficient tuning methods, but does not explicitly mention that which one is more stable / performant in the text. Therefore, the statement is not fully supported by the citation.[Response_End]\n
##\n
Statement: {statement}
References:\n{passages}\n
"""

editing_feedback_with_retrieval = """
We provide you with a question related to recent scientific literature, an answer from a strong language model, feedback on the answer, and relevant retrieved passages to address the feedback.

Your task is to incorporate the feedback to improve the answer by including new results or details from the retrieved passages (References:). When adding new information, avoid copying entire passages; instead, summarize the key information from the suggested papers to address the feedback. 
For instance, instead of copying the text from the original data, 'We found the empirical results X' to support the discussions without evidence in the original answer, say 'Former work found that X'. 
Only modify the parts mentioned in the feedback, keeping the rest of the answer intact.
Your improved answer should be marked with [Response_Start] and [Response_End].

Question: What are the advantages of retrieval-augmented LMs?
Answer: Retrieval-augmented LMs have been effective in various use cases, including reducing hallucinations [0] and enabling efficient adaptations to new data, such as temporal shifts [1]. Empirical results suggest they reduce hallucinations by 30% [2].
Feedback: The answer provides solid empirical results on hallucination reduction but lacks data on efficient adaptations. Please include empirical results for that aspect as well.
References:
[20] Title: Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs Text: Fine-Tuning vs. RAG: In MMLU and current events tasks, RAG consistently outperformed fine-tuning. RAG incorporates relevant context, unlike fine-tuning, which can lead to catastrophic forgetting. The performance of models like Orca2, fine-tuned via reinforcement learning, further demonstrates the advantages of RAG.
Edited Answer:
[Response_Start]Retrieval-augmented LMs have been effective in various use cases, including reducing hallucinations [0] and enabling efficient adaptations to new data, such as temporal shifts [1]. Empirical results suggest they reduce hallucinations by 30% [2]. Regarding efficient adaptations, studies show that RAG significantly outperforms fine-tuning in tasks like MMLU, offering better adaptation to new data without retraining the model [3].[Response_End]
"""
editing_with_retrieval_instance_prompt = \
    editing_feedback_with_retrieval +  "##\nQuestion: {question}\nAnswer:\n{answer}\nFeedback:\n{feedback}\nReferences:\n{retrieved_passages}\nEdited Answer:\n"


prompts_w_references_single_paper = ("Answer a question based on the following scientific paper. Your answer should sufficiently answer the question, citing specific paragraphs from the papers that provide full support for the statement. "
                                    "Your answer is likely to be one or more than one sentences."
                       "All of citation-worthy statements in your answer need to be supported by one of the references we provide as 'References:'. References should be formatted as [0], [1], [2], ..., [n]."
                       "Not all of the references are useful, and you only need to cite the references that support the sentences. "
                       "You only need to cite one paragraph for each citation worthy statement."
                       "Your answer should be marked as [Response_Start] and [Response_End]."
                       "Here's an example:\n##\n"
                       "References: \n{example_passages}"
                       "\nQuestion: {example_question}"
                       "\n[Response_Start]{example_answer}[Response_End]\nNow, please answer this question:\n##\n")
prompts_w_references_single_paper_zero_shot = ("Answer the following question based on a scientific paper. We provide a set of paragraphs from the paper, indicated as 'References'. Ensure that each answer includes citations that provide sufficient evidence to support it, using citation numbers for citation-worthy statements.\n")
generation_demonstration_prompts_single_paper = prompts_w_references_single_paper.format_map({"example_passages": example_passages_single_paper, "example_question": example_question_single_paper, "example_answer": example_answer_single_paper})
generation_instance_prompts_w_references_single_paper = generation_demonstration_prompts_single_paper + "References:\n {context}\nQuestion: {input}\n"
generation_instance_prompts_w_references_single_paper_zero_shot = prompts_w_references_single_paper_zero_shot + "\nReferences:\n{context}\nQuestion: {input}\n"


promts_w_references_single_paper_no_context = ("Answer a question based on the following scientific paper. "
                                    "Your answer is likely to be one or more than one sentences."
                       "Your answer should be marked as [Response_Start] and [Response_End]."
                       "Here's an example:\n##\n"
                       "\nQuestion: {example_question}"
                       "\n[Response_Start]{example_answer}[Response_End]\nNow, please answer this question:\n##\n")

generation_demonstration_prompts_single_paper_no_context = promts_w_references_single_paper_no_context.format_map({"example_question": example_question_single_paper, "example_answer": example_answer_single_paper_no_context})
generation_instance_prompts_w_references_single_paper_no_context = generation_demonstration_prompts_single_paper_no_context + "\nQuestion: {input}\n"


promts_w_references_summarization = ("Given an abstract of an academic paper and a set of passsages from relevant papers, generate a related work section summarizing relevant related work."
                                    "Not all of the passages are relevant, so please carefully read the passages and only use passages that are related."  
                                    "All of citation-worthy statements need to be supported by one of the references we provide as 'References' and appropriate citation numbers should be added at the last of the sentences."
                       "References should be formatted as [0], [1], [2], ..., [n]."
                       "Your answer should be marked as [Response_Start] and [Response_End]."
                       "Here's an example:\n##\n"
                       "References: \n{example_passages}"
                       "\nAbstract: {example_question}"
                       "\n[Response_Start]{example_answer}[Response_End]\nNow, please generate another related work given the following abstract.\n##\n")
generation_demonstration_summarization = promts_w_references_summarization.format_map({"example_passages": example_passages_summarization, "example_question": example_question_summarization, "example_answer": example_answer_summarization})
generation_instance_prompts_summarization = generation_demonstration_summarization + "References:\n {context}\nAbstract: {input}\n"


prompts_w_references_summarization_zero_shot = ("Given an abstract of an academic paper and a set of passages from relevant papers, generate a related work section summarizing relevant related work. All of citation-worthy statements need to be supported by one of the references we provide as 'References' and appropriate citation numbers should be added at the last of the sentences. References should be formatted as [0], [1], [2], ..., [n].\nReferences: {context}\nAbstract: {input}")

keyword_extraction_prompt = """
You are an expert at generating precise search queries for academic literature databases. Generate exactly four optimized search queries for the question below, formatted as comma-separated terms.

SEARCH STRATEGY:
• Primary query: Core concepts + key entities from question
• Specific query: Most specific technical terms + methodology 
• Broader query: Domain context + related concepts
• Alternative query: Synonyms + alternative terminology

OPTIMIZATION RULES:
• Use 2-6 words per query (optimal for both Semantic Scholar & OpenAlex)
• Include exact scientific terminology when available
• Combine key entities with biological/technical context
• Avoid stop words, articles, prepositions
• Use established field terminology and MeSH-like terms
• Include methodology terms when relevant (e.g., CRISPR, RNA-seq, proteomics)

BIOLOGICAL/MEDICAL FOCUS:
• Gene symbols in standard format (e.g., TP53, BRCA1)
• Disease names and classifications (e.g., lung cancer, adenocarcinoma)
• Cell lines with standard nomenclature (e.g., HeLa, MCF-7)
• Pathways using standard names (e.g., DNA repair, apoptosis)
• Techniques and assays (e.g., Western blot, flow cytometry)

Return only the four queries, wrapped between [Response_Start] and [Response_End].

EXAMPLES:

Question: How have prior work incorporated personality attributes to train personalized dialogue generation models?
[Response_Start]personalized dialogue generation models, personality-aware conversational AI, persona-conditioned language models, dialogue personalization techniques[Response_End]

Question: How do retrieval-augmented LMs perform well in knowledge-intensive tasks?
[Response_Start]retrieval-augmented language models, knowledge-intensive question answering, RAG performance evaluation, retrieval-based text generation[Response_End]

Question: What is the role of BRCA1 mutations in breast cancer development?
[Response_Start]BRCA1 mutations breast cancer, hereditary breast cancer genetics, DNA repair breast tumorigenesis, BRCA1 tumor suppressor function[Response_End]

Question: How does CRISPR-Cas9 work for gene editing in mammalian cells?
[Response_Start]CRISPR-Cas9 gene editing, mammalian genome editing, CRISPR mechanism action, programmable nuclease systems[Response_End]

Question: {question}
"""

relevance_judge_prompt = """
You are an expert relevance assessor for biomedical and life science literature. Your task is to evaluate whether a research paper can contribute meaningful insights to answer a specific scientific query. 

STEP 1: QUERY CLASSIFICATION
First, classify the query type based on these characteristics:

• COMPLEX queries: Novel mechanisms, therapeutic discovery, emerging biology, cutting-edge research, hypothesis generation, unexplored gene functions, novel disease mechanisms, emerging therapeutic targets
• SPECIFIC queries: Well-established pathways, well-defined biological processes, clear context-specific questions, established gene functions, known disease mechanisms, validated therapeutic approaches
• INTERDISCIPLINARY queries: Cross-field insights, translational research, systems biology, multi-omics integration, clinical-to-basic research translation, computational-experimental integration
• EXPLORATORY_MODE queries: Early-stage hypothesis generation, broad literature surveys, discovery-oriented research, open-ended questions, comparative studies across systems

Classification indicators:
- COMPLEX: Contains words like "novel", "emerging", "discovery", "unknown mechanism", "therapeutic potential", "new target"
- SPECIFIC: Contains well-defined biological entities, established pathways, specific protocols, validated approaches
- INTERDISCIPLINARY: Spans multiple biological domains, involves clinical translation, integrates different methodologies
- EXPLORATORY_MODE: Uses broad terms, asks open questions, seeks comprehensive understanding, comparative analysis

STEP 2: APPLY APPROPRIATE JUDGMENT STRATEGY

Based on the classified query type, apply the corresponding strategy:

FOR COMPLEX QUERIES - INCLUSIVE STRATEGY:
• Accept papers with ANY reasonable potential value, even if indirect
• Favor inclusion for novel insights, emerging mechanisms, or discovery potential
• Include methodological innovations and foundational studies
• Accept papers from related biological systems that could provide transferable insights

FOR SPECIFIC QUERIES - FOCUSED STRATEGY:
• Require moderate to high relevance with direct mechanistic or functional alignment
• Focus on papers directly addressing the specific biological entities or processes
• Accept contextual relevance only if it provides clear mechanistic insights
• Maintain stricter criteria for inclusion

FOR INTERDISCIPLINARY QUERIES - CONNECTIVE STRATEGY:
• Prioritize meaningful conceptual or mechanistic links across fields
• Include papers that bridge different biological domains or methodologies
• Accept translational studies and cross-system comparisons
• Value papers that provide methodological or conceptual frameworks

FOR EXPLORATORY_MODE QUERIES - MAXIMUM INCLUSIVITY STRATEGY:
• Accept papers with ANY potential connection or insight
• Include tangential relevance, especially novel perspectives or methods
• Maximize inclusion for comprehensive literature exploration
• Reject ONLY completely unrelated fields

INCLUSION CRITERIA (Accept if paper meets ANY of these, weighted by query type):

- DIRECT RELEVANCE (Highly Valuable for ALL query types)
   • Paper directly addresses the query's main topic with specific evidence
   • Contains the exact biological entities mentioned (genes, proteins, pathways, diseases, cell types)
   • Provides experimental data, mechanisms, or findings directly answering the question

- CONTEXTUAL RELEVANCE (Valuable, especially for COMPLEX and INTERDISCIPLINARY)
   • Studies related biological systems, pathways, or mechanisms that inform the query
   • Addresses analogous questions in related contexts (gene families, related diseases, similar pathways)
   • Provides complementary information that enhances understanding of the broader topic

- METHODOLOGICAL RELEVANCE (Valuable for COMPLEX and EXPLORATORY_MODE)
   • Demonstrates experimental techniques, assays, or computational approaches applicable to the query
   • Establishes analytical frameworks or validation methods relevant to the research area
   • Provides technological innovations that could advance the field of study

- FOUNDATIONAL RELEVANCE (Background value, especially for COMPLEX and EXPLORATORY_MODE)
   • Provides essential background knowledge for understanding the query context
   • Discusses fundamental principles, mechanisms, or discoveries in the relevant domain
   • Offers historical perspective or establishes theoretical foundations

- EXPLORATORY RELEVANCE (High value for EXPLORATORY_MODE and INTERDISCIPLINARY)
   • Reveals unexpected connections or cross-field insights
   • Presents novel hypotheses or theoretical frameworks that could inform the query
   • Demonstrates innovative approaches to related biological questions

EXCLUSION CRITERIA (Reject only if):
• For SPECIFIC queries: Papers with low relevance that don't provide clear mechanistic insights
• For ALL query types: Completely unrelated biological systems with zero transferable insights
• For ALL query types: Different contexts with no transferable insights for well-established questions
• For ALL query types: Purely technical methods papers without biological context or validation
• For COMPLEX/EXPLORATORY_MODE: Reject ONLY if paper is from completely unrelated field

DECISION LOGIC:
1. Classify the query type first
2. Apply the appropriate strategy based on classification
3. Make inclusion decision based on the strategy-specific criteria
4. When in doubt, refer to the strategy's default approach (lean toward inclusion for COMPLEX/EXPLORATORY_MODE, be more selective for SPECIFIC)

OUTPUT FORMAT:
Return exactly "True" or "False" (without quotes)

========

User Query: 
{query}

Paper Entity:
{paper_json}
"""


final_processing = """
Given the following answer to a question, reduce the repetitive discussions or less important details to make the answer more concise yet keep all of the important information relevat to the question kept. Do not loose any of the citation numbers originally provided by the answer. If there's any issue with organization, such as repetitive discussions, redundant information, disconnected discussions, please edit the text without changing the main content and do not loose any of the citation. If there's no issue, just copy the original text. Your answer should be marked as [Response_Start] and [Response_End].
Question: {question}\n
Answer: {answer}
"""