target_id_query_temp_1 = """
Which of these genes {candidate_genes} in {celltype} is the strongest candidate as a target for {disease}?
"""

target_id_query_temp_2 = """
Which of these genes {candidate_genes} in {celltype} is the strongest candidate in terms of target specificity to the cell type as a therapeutic target for {disease}? Use cosine similarity and target reference embedding to find the best.
"""

target_id_query_temp_3 = """
Which of these genes {candidate_genes} in {celltype} is the strongest candidate in terms of target specificity to the cell type as a therapeutic target for {disease}? Find the best one by compairing to a reference embedding of {disease} targets using cosine similarity.
"""

target_id_query_temp_4 = """
Among the genes {candidate_genes} in {celltype}, which one is the most specific to this cell type as a therapeutic target for {disease}? Create a reference embedding by averaging the embeddings of verified {disease} targets. Then, use cosine similarity to compare the candidates and identify the best match to the reference embedding.
"""

target_id_query_temp_5 = """
Which gene among {candidate_genes} shows the highest cell-type specificity in {celltype} for targeting {disease}?
"""

target_id_open_end_query_temp_1 = """
We have {celltype} from a patient with {disease}. What is the therapeutic potential of targeting {candidate_genes} in {celltype} to treat {disease}?
"""


target_id_instruction_1 = """ Identify the verified therapeutic targets for {disease} and retrieve their cell-type–specific embeddings from {scfm}. Determine which of the {cell_type}-activated genes in {scfm} overlap with these targets; this overlapping set constitutes the cell-type–specific targets for {disease}. Compute their average embedding as a reference, then calculate the cosine similarity between this reference and each individual target. If one gene shows a markedly higher similarity than the rest, nominate it as the top candidate and supply a concise, evidence-based rationale for its biological relevance.
"""

target_id_instruction_2 = """ Begin by identifying the verified therapeutic targets for {disease} and retrieving their corresponding cell-type-specific gene embeddings from {scfm}. Next, examine the {cell_type}-specific activated genes available in {scfm} and determine their overlap with the verified disease targets—this subset represents the cell-type-specific target genes for {disease}. 

To establish a reference embedding, compute the average embedding of these overlapping genes. Then, assess the target gene by calculating its cosine similarity to the reference embedding. If a the gene exhibits a significantly high similarity score, designate it as the target candidate and find the biological justification for its relevance with scientific reasoning.
"""




REPHRASE_TEMPLATE = """
You are a professional {role}. Your task is to refine the given question into a clear, precise, and scientifically sound inquiry while maintaining a realistic tone that a {role} would use in daily practice. Ensure that the core information is preserved, but enhance clarity, specificity, and readability as necessary.
------
Given Question:

{task_instruction_template}

------
Refinement Guidelines:
	1.	Maintain the core scientific intent and key details.
	2.	Use terminology and phrasing that align with how {role}s naturally frame such inquiries.
	3.	Ensure clarity and precision while keeping the question concise and focused.
	4.	Retain the requirement for a single best candidate gene as the output.

IMPORTANT: Return ONLY the refined question text. Do not include any labels, prefixes, or additional text like "Refined Question:" or "Answer:". Just return the refined question directly.
"""

IMMUNE_REPHRASE_TEMPLATE = """
Your task is to refine the given question into a clear, precise, and scientifically sound inquiry while maintaining a realistic tone that a {role} would use. Ensure that the core information is preserved, but enhance clarity, specificity, and readability as necessary.
------
Given Question:

{task_instruction_template}

------
Refinement Guidelines:
	1.	Maintain the core scientific intent and key details (e.g., file path).
	2.	Use terminology and phrasing that align with how {role}s naturally frame such inquiries.

Your refined question should be consistent, realistic and naturally phrased as a {role} would articulate it. 

IMPORTANT: Return ONLY the refined question text. Do not include any labels, prefixes, or additional text like "Refined Question:" or "Answer:". Just return the refined question directly.
"""

EXPERIMENT_INSTRUCTION_REPHRASE_TEMPLATE = """
You are a professional {role}. Your task is to paraphrase the given instruction into a clear, precise, and scientifically sound instruction while maintaining a realistic tone that a {role} would use in daily practice. Ensure that the core information is preserved, but enhance clarity, specificity, and readability as necessary.
------
Given Instruction:

{task_instruction_template}
"""



immune_query_temp_1 = """I have a {sex} patient with {disease}, and their mRNA TPM expression was measured from a {tissue} biopsy to assess the tumor immune microenvironment. The patient has a tumor mutational burden (TMB) of {tmb}. I am considering prescribing {treatment}. Based on these factors, is the patient likely to respond to this therapy?
"""

ICI_FILE_PATH = """ The patient's transcriptomic profile (TME) is stored in the file {tpm_path}.pkl. """

immune_instruction_1 = """"""


immune_query_temp_2 = """My {race} {sex} patient has {disease}. I have a transcriptomic profile of the patient's tumor immune microenvironment from {tissue} biopsy. The patient has TMB of {tmb}. Will the patient respond to {treatment} therapy? 
"""

immune_instruction_2 = """ Analyze the patient's tumor microenvironment transcriptomic profile, with an emphasis on the expression of exhausted T cells, cytotoxic T cells, monocytes, and macrophages. Provide supported reasoning to determine whether the patient is likely to be a responder"""



immune_query_temp_3 = """ I have processed mRNA TPM data representing a patient's tumor immune microenvironment for {disease}, and I would like to assess the predicted responsiveness of this patient to {treatment}. Below are the detailed patient attributes:

- Sex: {sex}
- Race: {race}
- Tissue: {tissue}
- Disease: {disease}
- Tumor Mutational Burden (FMOne mutation burden per MB): {tmb}

Please provide a conclusion on whether the patient is classified as a responder or non-responder, along with the supporting reasoning.
"""


immune_instruction_3 = """ Load the transcriptomic data from the specified pickle file and apply Compass to predict the patient's likelihood of responding to ICI therapy. From the Compass output, extract the top five most relevant immune-related concepts, prioritizing those with the highest significance scores. Merge these insights with the patient's metadata to construct a comprehensive summary of their immune landscape. Use the Compass predicted responder status as a key signal, alongside the immune concept data, to guide downstream scientific reasoning. Generate a detailed reasoning instructions that incorporate this prediction to determine, with justification, whether the patient is a likely responder or non-responder to ICI therapy.
"""


immune_query_temp_4 = """ For my {race} {sex} patient diagnosed with {disease}, I obtained a transcriptomic profile of the tumor immune microenvironment from a {tissue} biopsy. The patient exhibits a TMB of {tmb}. Based on these parameters, what is the likelihood of a positive response to {treatment} therapy?"""



immune_instruction_4 = """ Evaluate the tumor microenvironment transcriptomic profile using COMPASS to predict treatment response. Focus on immune deficiency markers—NK cells, exhausted T cells, general B cells, and plasma cells. Use these markers alongside the COMPASS prediction to provide evidence-based reasoning on whether the patient is likely to respond to treatment. Support your analysis with relevant transcriptomic features and their implications for immune activity.
"""


immune_query_temp_5 = """ I have a {race} {sex} patient diagnosed with {disease}. The transcriptomic profile of their tumor immune microenvironment was obtained from a {tissue} biopsy, revealing a TMB of {tmb}. Can you assess whether this patient is likely to respond to {treatment} therapy?
"""


immune_instruction_5 = """ Review transcriptomic data and run Compass to predict ICI therapy response. Evaluate the Compass output for immune efficiency markers - macrophage, IFNG pathway, genome integrity, cell proliferation, and cytotoxic T cells. Integrate these findings with the Compass prediction and patient metadata for a comprehensive immune landscape summary. Finally, detailed reasoning instructions will be generated that use the predicted responder status and immune concept data to justify whether the patient is likely a responder or non-responder to ICI therapy.
"""


immune_query_temp_6 = """ I have processed mRNA TPM data representing a patient's tumor immune microenvironment for {disease}, and I would like to assess the predicted responsiveness of this patient to {treatment}. Below are the detailed patient attributes:

- Sex: {sex}
- Race: {race}
- Tissue: {tissue}
- Disease: {disease}
- Tumor Mutational Burden (FMOne mutation burden per MB): {tmb}

Please provide a conclusion on whether the patient is classified as a responder or non-responder, along with the supporting reasoning.
"""

immune_instruction_6 = """ Evaluate the tumor microenvironment transcriptomic profile using COMPASS to predict treatment response. Focus on immune deficiency markers (NK cells, exhausted T cells and bcell general) and immune enhancement markers (macrophages, IFNG pathway, and genome integrity). Combine these markers with the COMPASS prediction to provide evidence-based reasoning on the patient's likely response to treatment, supported by relevant transcriptomic features and their implications for immune activity.
"""


RESPONSE_CHECKER = """Evaluate whether the given response sufficiently addresses the user's query. Answer 'Yes' or 'No.'
---
User Query: {query}
---
Given Response: {response}
---
Output format: (yes/no)
"""


sl_query_tissue_openend = """
You are a biomedical researcher with expertise in genetic interactions. Your task is to simulate a realistic query that a biologist might ask regarding the potential synthetic genetic interaction between two mutated genes, given the following context:

- Mutated Gene A: [Gene A]
- Mutated Gene B: [Gene B]
- Synonyms for Gene A: [Gene A synonyms]
- Synonyms for Gene B: [Gene B synonyms]
- Cell Line: [Cell Line]
{addition_context_temp}

The biologist's query should:
	1.	Inquire about the potential synthetic genetic interaction between Gene A and Gene B within the specified biological context. If such an interaction exists, request the specific name of the interaction.
	2.	Frame the context in terms of tissue and cancer type, avoiding direct mention of cell lines. For example, instead of referencing the "HeLa" cell line, refer to "cervical epithelium in cervical adenocarcinoma".
	3.	Be structured as a step-by-step, open-ended question that encourages systematic analysis of the potential interaction.
	4.	Incorporate one synonym from the provided list for each mutated gene when formulating the query. Use only one synonym per gene to maintain clarity. For example, use "BRCC1" rather than "BRCA1 (also known as BRCC1 or BROVCA1).""

Example Input:
- Mutated Gene A: CSK
- Mutated Gene B: GATAD1
- Synonyms for Gene A: None
- Synonyms for Gene B: ['CMD2B', 'ODAG', 'RG083M05.2']
- Cell Line: T47D

Example Output:

"In the context of luminal A subtype breast cancer, considering concurrent mutations in CSK and GATAD1 in the breast epithelium, please systematically evaluate their potential synthetic genetic interaction by addressing the following points step-by-step:
	1.	CSK mutation: Assess how a mutation in CSK might influence cellular signaling pathways, particularly those related to the regulation of gene expression or protein stability.
	2.	GATAD1 mutation: Evaluate how a mutation in GATAD1 could impact chromatin remodeling or transcriptional regulation mechanisms.
Based on your analysis of these individual impacts, identify the most likely type of genetic interaction resulting from simultaneous mutations in both genes."

Now, generate a realistic query based on the following input:
- Mutated Gene A: {gene_a}
- Mutated Gene B: {gene_b}
- Synonyms for Gene A: {synonyms_a}
- Synonyms for Gene B: {synonyms_b}
- Cell Line: {cell_line}
{addition_context_sample}
"""



sl_query_lineage_openend = """
Your task is to generate a realistic query that a biologist would ask about the synthetic genetic interaction between two mutated genes, using the provided context:
    - Mutated Gene A: [Gene A]
    - Mutated Gene B: [Gene B]
    - Cell Line: [Cell Line]


Guidelines:
    1. Avoid mentioning any specific type of synthetic genetic interaction in the query, or hinting at the expected type of interaction.
    2. Ask clearly for what would be the synthetic genetic interaction with a tone that a biologist would use in daily practice.

Example Input:
    - Mutated Gene A: CSK
    - Mutated Gene B: GATAD1
    - Cell Line: T47D

Example Output:
    "We have introduced a double mutation in CSK and GATAD1 in T47D breast cancer cell line. What would be the synthetic genetic interaction caused by this concurrent mutation if exist?"

Now, generate a realistic query based on the following input (return only the query, no other text):
    - Mutated Gene A: {gene_a}
    - Mutated Gene B: {gene_b}
    - Cell Line: {cell_line}
"""


sl_query_tissue_multi = """
You are a biomedical researcher with expertise in genetic interactions. Your task is to simulate a realistic query that a biologist might ask regarding the potential synthetic genetic interaction between two mutated genes, given the following context:

- Mutated Gene A: [Gene A]
- Mutated Gene B: [Gene B]
- Cell Line: [Cell line]

Your generated biologist query should:
1.	Clearly ask whether mutations in Gene A and Gene B could result in a synthetic lethal interaction within the specified cell line.
2.	Be phrased as an open-ended, step-by-step question that encourages in-depth reasoning about the underlying biological mechanisms.

Example Input:
	- Mutated Gene A: BRCA1
	- Mutated Gene B: PARP1
	- Cell Line: HeLa

Example Output:

"Given concurrent mutations in BRCC1 and PARP1, could this lead to a synthetic lethal interaction within cervical carcinoma tissues? If so, please elucidate the underlying mechanisms step by step."

⸻

Now, generate a realistic biologist-style query based on the following input:

- Mutated Gene A: {gene_a}
- Mutated Gene B: {gene_b}
- Cell Line: {cell_line}
"""


sl_query_lineage_multi = """
Your task is to generate a realistic query that a biologist would ask regarding the cell viability based on the synthetic mutation of two genes, using the provided context:

- Mutated Gene A: [Gene A]
- Mutated Gene B: [Gene B]
- Cell Line: [Cell Line]


Example Input:
- Mutated Gene A: BRCA1
- Mutated Gene B: PARP1
- Cell Line: HeLa

Example Output:

"We have introduced a double mutation in BRCA1 and PARP1 in HeLa cell line. Will the cell be viable despite the simultaneous perturbation of both genes?"

⸻

Now, generate a realistic biologist-style query based on the following input:

- Mutated Gene A: {gene_a}
- Mutated Gene B: {gene_b}
- Cell Line: {cell_line}
"""






sl_instruction_1 = ""

sl_instruction_h2 = """ Analyze the interaction network within a tissue-specific context, focusing on the relationship between the two genes. Identify the shortest connected path between them, if available, and gather information on shared biological processes and pathways. Use this enriched data to perform a detailed reasoning, and analyze the most potential genetic interaction."""


# [Genetic Interaction] Enrichr only
sl_instruction_e1 = """ Use the Enrichr API to retrieve up to 10 enriched pathways that are most relevant to cell viability for the provided pair of mutated genes. Construct a concise summary with those pathways and using this short summary as input, apply step-by-step reasoning to predict the most likely genetic interaction that could result from the concurrent mutation of both genes."""


# [SL prediction] Enrichr only
sl_instruction_e2 = """ Utilize Enrichr to identify the top 10 enriched pathways associated with cell viability for the specified pair of mutated genes. Based on these enriched pathways, construct a comprehensive statement summarizing the potential biological implications. Subsequently, employ detailed, step-by-step scientific reasoning to assess the likelihood of a genetic interaction between the two genes, and predict the most probable type of interaction, if any."""



# [SL prediction] DepMap + Enrichr
sl_instruction_e3 = """ Using DepMap data and Enrichr API, first obtain both the enrichment analysis results—detailing the top 5 key functional pathways and biological processes linked to these genes—and the correlation metrics (including the correlation coefficient and significance levels) that reflect the pair's co-dependency. Based on that, using this functional summary as the basis, apply a step-by-step reasoning process to predict the cell viability, with a particular focus on evaluating whether their concurrent mutation is expected to lead to synthetic lethality."""


# [SL prediction] DepMap only
sl_instruction_e4 = """ Using DepMap data, first obtain the correlation metrics (including the correlation coefficient and significance levels) that reflect the pair's co-dependency. Based on that, using this functional summary as the basis, apply a step-by-step reasoning process to predict the cell viability, with a particular focus on evaluating whether their concurrent mutation is expected to lead to synthetic lethality."""


# [SL prediction] DepMap -> Enrichr
sl_instruction_e5 = """ Use DepMap data to retrieve correlation metrics reflecting the co-dependency of the gene pair on cell viability. Next, perform pathway enrichment analysis with Enrichr to identify whether pathways associated with cell viability are significantly enriched and could be impacted by the gene pair. Synthesize the DepMap and Enrichr results, evaluate whether the combined perturbation of these genes is likely to induce a significant effect on cell viability, and find literature support if exist."""


# [Genetic Interaction] DepMap -> Enrichr 
sl_instruction_e6 = """ Using DepMap data, first retrieve the correlation metrics—such as the correlation coefficient and significance levels—that indicate the co-dependency of the provided gene pair on cell viability. Validate that the pair shows a significant impact on cell viability. Additionaly, perform enrichment analysis using Enrichr to identify up to the top 5 enriched pathways that are most relevant to cell viability. If pathways exist, form a statement based on the DepMap metrics and pathway insights and using this statement as input, provide a detailed, step‑by‑step scientific reasoning to predict the most likely genetic interaction. """


TARGETID_REASON_CHECK = """
From the statement, extract the single strongest target gene being proposed/suggested. If a single gene is clearly recommended as top/primary/most promising, return that gene’s symbol verbatim. If no unambiguous gene is proposed, output a label. Return only the target gene name or category label (no extra text).

Statement:
{reasoning_result}

Categories:
    • Abstain: If the text indicates that evidence is insufficient or ambiguous (“no studies found,” “can’t determine,” “further research needed”) without evaluating each candidate in detail.
    • None: If the text explicitly evaluates and disqualifies each candidate—showing why each fails to meet the context-specific criteria.
    • Failed: If the paragraph refuses, reports inability, or returns an error (e.g., "I can't help", "failed", "none" with no reasoning).
"""


SL_REASON_CHECK = """Given a reasoning paragraph, determine which category or categories below it fits—based on whether it concludes that synthetic lethality (SL) represents a (potential) genetic interaction between two genes. Return only the exact category name(s), with no additional commentary.

----
Reasoning Paragraph:
{reasoning_result}

----
Categories:
- Synthetic lethality: Select this category if the paragraph clearly concludes or strongly supports that the combined perturbation (loss/inhibition/KO) of the two genes reduces viability, and concludes that synthetic lethality (SL). Signals include: “synthetic lethal/sickness,” “double knockout is lethal,” “co-inhibition is lethal/toxic”.
- Non-SL: Select if the paragraph argues that synthetic lethality is unlikely.
- Abstain: Select if the paragraph is ambiguous, inconclusive, or does not clearly support or refute synthetic lethality.
- Failed: Select if the paragraph refuses, reports inability, or returns an error (e.g., "I can't help", "failed", "none" with no reasoning).

Provide only the name(s) of the applicable category or categories.
"""


IMMUNE_ANS_CHECK = """
Determine which category best describes the statement below. Output only the category label (no extra text):

Statement:
{reasoning_result}

Categories:
    - R: A positive verdict that response/benefit is likely (e.g., "likely/moderately likely to respond", "responder", "clinical benefit expected").
    - NR: A negative verdict that response is unlikely (e.g., "unlikely to respond", "non-responder", "no benefit", "resistant").
    - Abstain: Ambiguous or mixed evidence with no overall verdict; hedged assessments without a clear lean; requests more information.
    - Failed: The paragraph refuses, reports inability, or returns an error (e.g., "I can't help", "failed", "none" with no reasoning).
"""
