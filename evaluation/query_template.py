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

target_id_query_temp = """
Which gene among {candidate_genes} shows the highest cell-type specificity in {celltype} for targeting {disease}?
"""

target_id_instruction = """ Identify the verified therapeutic targets for {disease} and retrieve their cell-type–specific embeddings from {scfm}. Determine which of the {cell_type}-activated genes in {scfm} overlap with these targets; this overlapping set constitutes the cell-type–specific targets for {disease}. Compute their average embedding as a reference, then calculate the cosine similarity between this reference and each individual target. If one gene shows a markedly higher similarity than the rest, nominate it as the top candidate and supply a concise, evidence-based rationale for its biological relevance.
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


# Synthetic Lethality Prediction
sl_instruction_default = """ Use DepMap data to retrieve correlation metrics reflecting the co-dependency of the gene pair on cell viability. Next, perform pathway enrichment analysis with Enrichr to identify whether pathways associated with cell viability are significantly enriched and could be impacted by the gene pair. Synthesize the DepMap and Enrichr results, evaluate whether the combined perturbation of these genes is likely to induce a significant effect on cell viability, and find literature support if exist."""


immune_query_default = """I have a {sex} patient with {disease}, and their mRNA TPM expression was measured from a {tissue} biopsy to assess the tumor immune microenvironment. The patient has a tumor mutational burden (TMB) of {tmb}. I am considering prescribing {treatment}. Based on these factors, is the patient likely to respond to this therapy?
"""

ICI_FILE_PATH = """ The patient's transcriptomic profile (TME) is stored in the file {tpm_path}.pkl. """

immune_no_instruction = """"""


immune_query_temp_a = """ I have processed mRNA TPM data representing a patient's tumor immune microenvironment for {disease}, and I would like to assess the predicted responsiveness of this patient to {treatment}. Below are the detailed patient attributes:

- Sex: {sex}
- Race: {race}
- Tissue: {tissue}
- Disease: {disease}
- Tumor Mutational Burden (FMOne mutation burden per MB): {tmb}

Please provide a conclusion on whether the patient is classified as a responder or non-responder, along with the supporting reasoning.
"""


immune_instruction_a = """ Load the transcriptomic data from the specified pickle file and apply Compass to predict the patient's likelihood of responding to ICI therapy. From the Compass output, extract the top five most relevant immune-related concepts, prioritizing those with the highest significance scores. Merge these insights with the patient's metadata to construct a comprehensive summary of their immune landscape. Use the Compass predicted responder status as a key signal, alongside the immune concept data, to guide downstream scientific reasoning. Generate a detailed reasoning instructions that incorporate this prediction to determine, with justification, whether the patient is a likely responder or non-responder to ICI therapy.
"""


immune_query_temp_b = """ For my {race} {sex} patient diagnosed with {disease}, I obtained a transcriptomic profile of the tumor immune microenvironment from a {tissue} biopsy. The patient exhibits a TMB of {tmb}. Based on these parameters, what is the likelihood of a positive response to {treatment} therapy?"""



immune_instruction_b = """ Evaluate the tumor microenvironment transcriptomic profile using COMPASS to predict treatment response. Focus on immune deficiency markers—NK cells, exhausted T cells, general B cells, and plasma cells. Use these markers alongside the COMPASS prediction to provide evidence-based reasoning on whether the patient is likely to respond to treatment. Support your analysis with relevant transcriptomic features and their implications for immune activity.
"""




RESPONSE_CHECKER = """Evaluate whether the given response sufficiently addresses the user's query. Answer 'Yes' or 'No.'
---
User Query: {query}
---
Given Response: {response}
---
Output format: (yes/no)
"""


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
