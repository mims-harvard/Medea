from tqdm import tqdm
from nltk import sent_tokenize
from . import instructions
from .gpt_utils import chat_completion
import requests
import time
import re




# nlp = spacy.load('en_core_web_sm')

def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")
    
def rerank_paragraphs_bge(query, paragraphs, reranker, norm_cite=False, start_index=0, use_abstract=False, batch_size=50):
    """
    Re-rank paragraphs using a BGE-based reranker, optionally adjusting scores based on citation counts.

    Parameters:
        query (str): The query text.
        paragraphs (list of dict): List of paragraph dictionaries; each must contain a "text" field.
        reranker (object): An object with a `compute_score` method that accepts a list of [query, text] pairs.
        norm_cite (bool): If True, adjust scores by adding a normalized citation count.
        start_index (int): Starting index for ranking (currently unused, reserved for future extensions).
        use_abstract (bool): If True, include the abstract (and title) in the text used for scoring.

    Returns:
        new_orders (list): Reordered list of paragraph dictionaries, ranked from highest to lowest score.
        result_dic (dict): Mapping from original paragraph index to computed score.
        id_mapping (dict): Mapping from new order index to original paragraph index.
    """
    # Filter out paragraphs with missing text
    valid_paragraphs = [p for p in paragraphs if p.get("text") is not None]

    # Prepare text input for each paragraph based on the use_abstract flag
    if use_abstract:
        paragraph_texts = [
            (f"{p.get('title', '')}\n{p.get('abstract', '')}\n{p['text']}"
             if p.get("title") is not None and p.get("abstract") is not None else p["text"])
            for p in valid_paragraphs
        ]
    else:
        paragraph_texts = [
            (f"{p.get('title', '')} {p['text']}" if p.get("title") is not None else p["text"])
            for p in valid_paragraphs
        ]

    # Compute scores using the reranker; each input is a pair [query, paragraph_text]
    scores = reranker.compute_score([[query, text] for text in paragraph_texts], batch_size=batch_size)

    # Wrap a single float score in a dictionary; otherwise, enumerate the scores
    if isinstance(scores, float):
        result_dic = {0: scores}
    else:
        result_dic = {idx: score for idx, score in enumerate(scores)}

    # Optionally adjust scores based on normalized citation counts, if available
    if norm_cite:
        # Extract citation counts for paragraphs that include them
        citation_counts = [p["citation_counts"] for p in valid_paragraphs if p.get("citation_counts") is not None]
        if citation_counts:
            max_citations = max(citation_counts)
            # Update scores by adding the normalized citation count to each paragraph's score
            for idx, p in enumerate(valid_paragraphs):
                if p.get("citation_counts") is not None:
                    result_dic[idx] += p["citation_counts"] / max_citations

    # Sort paragraphs by score in descending order
    sorted_scores = sorted(result_dic.items(), key=lambda x: x[1], reverse=True)

    # Reorder paragraphs and build an index mapping (new index -> original index)
    new_orders = []
    id_mapping = {}
    for new_idx, (orig_idx, _) in enumerate(sorted_scores):
        new_orders.append(valid_paragraphs[orig_idx])
        id_mapping[new_idx] = orig_idx

    return new_orders, result_dic, id_mapping



def search_paper_via_query(query, max_paper_num=2, attempt=3, minCitationCount=5):
    if "Search queries: " in query:
        query = query.split("Search queries: ")[1]
    
    paper_field_collection = 'title,year,abstract,authors.name,citationCount,year,url,externalIds,publicationVenue'
    query_params = {
        "query": query, 
        "limit": max_paper_num, 
        "minCitationCount": minCitationCount, 
        "sort": "citationCount:desc", 
        "fields": paper_field_collection
    }
    # api_key = "19dnRruThD7a8AusEjLna3dgPtPx2vlH8bCsfite"
    # Send the API request

    while attempt > 0:
        response = requests.get('https://api.semanticscholar.org/graph/v1/paper/search', params=query_params)
        time.sleep(2)

        if response.status_code == 200:
            response_data = response.json()
            break
        # Process and print the response data as needed
        else:
            attempt -= 1
            response_data = None
            # print(f"Request failed with status code {response.status_code}: {response.text}")
            time.sleep(5)
    # except:
        # response_data = None
    if response_data is None or len(response_data) == 0 or "data" not in response_data:
        # print(f"retrieval failed: {response_data}")
        return None
    else:
        return response_data["data"]



def create_prompt_with_llama3_format(prompt, system_message="You are a helpful AI assistant for scientific literature review. Please carefully follow user's instruction and help them to understand the most recent papers."):
    if system_message is not None:
        formatted_text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{0}<|eot_id|>".format(system_message)
    else:
        formatted_text = "<|begin_of_text|>"
    formatted_text += "<|start_header_id|>user<|end_header_id|>\n\n" + prompt + "<|eot_id|>"
    formatted_text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return formatted_text

   
class OpenScholar(object):
    def __init__(
            self, 
            model, 
            tokenizer, 
            client_llm=None, 
            use_contexts=True, 
            top_n=8, 
            reranker=None, 
            min_citation=None, 
            norm_cite=False, 
            ss_retriever=False
        ):
        self.model = model
        self.tokenizer = tokenizer
        self.client_llm = client_llm
        self.top_n = top_n
        self.no_retrieval = not use_contexts
        self.reranker = reranker
        self.min_citation = min_citation
        self.norm_cite = norm_cite
        self.ss_retriever = ss_retriever
        self.use_contexts = use_contexts

    # Reranking: We rerank passages based on the LMs' predictions on how useful passages are.
    def process_ranking_results(self, result):
        ratings = {int(match.group(1)): int(match.group(2)) for match in re.finditer(r'\[(\d+)\] Rating: (\d)', result)}
        return ratings

    def reranking_passages_cross_encoder(self, item, use_abstract=False):
        
        if self.min_citation is not None:
            ctx_above_threshold = [p for p in item["ctxs"] if "citation_counts" in p and p["citation_counts"] >= self.min_citation]
            if len(ctx_above_threshold) > self.top_n:
                item["ctxs"] = ctx_above_threshold
                # print("after filtering -- number of ctxs: {0}".format(len(item["ctxs"])))
                
        reranked_contexts, sorted_results, id_mapping = rerank_paragraphs_bge(
            item["input"], 
            item["ctxs"], 
            self.reranker, 
            norm_cite=self.norm_cite, 
            use_abstract=use_abstract
        )
        return reranked_contexts, sorted_results, id_mapping
    
    def reranking_passages_cross_encoder_supplemental(self, item, passages):
        
        if self.min_citation is not None:
            ctx_above_threshold = [p for p in passages if "citation_counts" in p and p["citation_counts"] >= self.min_citation]
            if len(ctx_above_threshold) > self.top_n:
                passages = ctx_above_threshold
                # print("after filtering -- number of ctxs: {0}".format(len(passages)))
        
        reranked_contexts, sorted_results, id_mapping = rerank_paragraphs_bge(
            item["input"], 
            passages, 
            self.reranker, 
            norm_cite=False, 
            start_index=len(item["ctxs"])
        )

        return reranked_contexts, sorted_results, id_mapping
    
    def retrieve_keywords(self, question):
        prompt = [instructions.keyword_extraction_prompt.format_map({"question": question})]
        
        # Use default client
        raw_output = chat_completion(prompt[0], model=self.client_llm)
        outputs = raw_output
        raw_output = [t.split("[Response_End]")[0]  for t  in outputs.split("[Response_Start]") if "[Response_End]" in t][0] if "[Response_End]" in outputs else outputs
 
        queries = raw_output.split(", ")[:3]
        queries = [query.replace("Search queries: " , "") for query in queries if len(query) > 0]
        return queries

    # Generation: Generate output based on query, passages
    def generate_response(self, item, max_tokens=3000, llama3_chat=False,  task_name="default", zero_shot=False):
        ranked_results = {}
        # print("zero-shot?: {}".format(zero_shot))
        if self.use_contexts is False:
            ctxs = []
            # support more task
            if task_name in instructions.task_instructions:
                if zero_shot is True:
                    input_query = instructions.task_instructions[task_name][0] + instructions.task_instructions[task_name][1] + item["input"]
                else:
                    demonstration = instructions.demonstrations[task_name]
                    input_query = instructions.task_instructions[task_name][0] + demonstration + instructions.task_instructions[task_name][1] + item["input"]
            if  task_name == "single_qa":
                input_query = instructions.generation_instance_prompts_w_references_single_paper_no_context.format_map({"input": item["input"]})
            else:
                if zero_shot is True:
                    input_query = instructions.generation_instance_prompts_w_references_zero_shot.format_map({"context": ctxs, "input": item["input"]})
                else:
                    input_query = instructions.generation_instance_prompts_w_references.format_map({"context": ctxs, "input": item["input"]})
            item["final_passages"] = ctxs
        else:
            ctxs = ""
            for doc_idx, doc in enumerate(item["ctxs"][:self.top_n]):
                if "title" in doc and len(doc["title"]) > 0:
                    ctxs += "[{0}] Title: {1} Text: {2}\n".format(doc_idx, doc["title"], doc["text"])
                else:
                    ctxs += "[{0}] {1}\n".format(doc_idx,  doc["text"])
            item["final_passages"] = ctxs
            
            if task_name =="summarization":
                if zero_shot is True:
                    input_query = instructions.prompts_w_references_summarization_zero_shot.format_map({"context": ctxs, "input": item["input"]})
                else:
                    input_query = instructions.generation_instance_prompts_summarization.format_map({"context": ctxs, "input": item["input"]})
            elif task_name == "single_qa":
                if zero_shot is True:
                    input_query = instructions.generation_instance_prompts_w_references_single_paper_zero_shot.format_map({"context": ctxs, "input": item["input"]})
                else:
                    input_query = instructions.generation_instance_prompts_w_references_single_paper.format_map({"context": ctxs, "input": item["input"]})
            
            elif task_name in instructions.task_instructions:
                task_instruction = instructions.task_instructions[task_name][0]
                instance_header = instructions.task_instructions[task_name][1]
                if zero_shot is True:
                    input_query = "{0}\nReferences:\n{1}\n{2}{3}".format(task_instruction, ctxs, instance_header, item["input"])
                else:
                    demonstration = instructions.demonstrations[task_name]
                    input_query = "{0}{1}\nReferences:\n{2}\n{3}{4}".format(task_instruction, demonstration, ctxs, instance_header, item["input"])
                    
            else:
                if zero_shot is True:
                    input_query = instructions.generation_instance_prompts_w_references_zero_shot.format_map({"context": ctxs, "input": item["input"]})
                else:
                    input_query = instructions.generation_instance_prompts_w_references.format_map({"context": ctxs, "input": item["input"]})

        if llama3_chat is True:
            input_query = create_prompt_with_llama3_format(input_query)
            
        outputs = chat_completion(input_query, model=self.client_llm)
        raw_output = [t.split("[Response_End]")[0] for t in outputs.split("[Response_Start]") if "[Response_End]" in t][0] if "[Response_End]" in outputs else outputs

        if "References:" in raw_output:
            raw_output = raw_output.split("References:")[0]
        item["output"] = raw_output
        return raw_output, ctxs

    # Feedback: send feedback on model' predictions.
    def process_feedback(self, response):
        feedbacks_and_questions = re.findall(r'Feedback: (.*?)(?:Question: (.*?))?\n', response)
        ratings = [(feedback.strip(), question.strip() if question else "") for feedback, question in feedbacks_and_questions]
        return ratings

    def get_feedback(self, item, llama3_chat):
        input_query = instructions.feedback_example_instance_prompt.format_map(
            {
                "question": item["input"], 
                "passages": item["final_passages"], 
                "answer": item["output"]
            }
        )
        # TODO: check if the llama3 chat format is helpful or not. 
        if llama3_chat is True:
            input_query = create_prompt_with_llama3_format(input_query)
        
        outputs = chat_completion(input_query, model=self.client_llm)
        raw_output = [t.split("[Response_End]")[0]  for t  in outputs.split("[Response_Start]") if "[Response_End]" in t][0] if "[Response_End]" in outputs else outputs
        feedbacks = self.process_feedback(raw_output)
        return feedbacks

    def edit_with_feedback(self, item, feedback, max_tokens=3000, llama3_chat=False):
        input_query = instructions.editing_instance_prompt.format_map({"question": item["input"], "passages": item["final_passages"], "answer": item["output"], "feedback": feedback})
        
        # TODO: check if the llama3 chat format is helpful or not. 
        if llama3_chat is True:
            input_query = create_prompt_with_llama3_format(input_query)

        outputs = chat_completion(input_query, model=self.client_llm)
        raw_output = [t.split("[Response_End]")[0]  for t  in outputs.split("[Response_Start]") if "[Response_End]" in t][0] if "[Response_End]" in outputs else outputs
        return raw_output

    def edit_with_feedback_retrieval(self, item, feedback, passages, passage_start_index, max_tokens=2000, llama3_chat=False):
        processed_passages = ""
        for doc_idx, doc in enumerate(passages[:self.top_n]):
            if "title" in doc and len(doc["title"]) > 0:
                processed_passages += "[{0}] Title: {1} Text: {2}\n".format(passage_start_index+doc_idx, doc["title"], doc["text"])
            else:
                processed_passages += "[{0}] {1}\n".format(passage_start_index+doc_idx + len(item["ctxs"]), doc["text"])

        input_query = instructions.editing_with_retrieval_instance_prompt.format_map({"question": item["input"], "retrieved_passages": processed_passages, "answer": item["output"], "feedback": feedback})
        if llama3_chat is True:
            input_query = create_prompt_with_llama3_format(input_query)
                
        outputs = chat_completion(input_query, model=self.client_llm)
        raw_output = [t.split("[Response_End]")[0]  for t in outputs.split("[Response_Start]") if "[Response_End]" in t][0] if "[Response_End]" in outputs else outputs
        return raw_output

    def insert_attributions_posthoc_paragraph(self, item, llama3_chat=False):
        text = item["output"]
        if "final_passages" in item:
            passages = item["final_passages"] 
        else:
            ctxs = item["ctxs"]
            passages = ""
            for idx, p in enumerate(ctxs):
                passages += "[{0}] {1}\n".format(idx, p)

        # print(text)
        sentences = text.split("\n")
        # print(sentences)
        # post process sentences 
        updated_sentences = []
        post_hoc_sentence = {}

        for s_index, statement in enumerate(sentences):
            if len(statement) < 10:
                if len(updated_sentences) > 0 and len(statement) > 0 and statement[0] == "[":
                    updated_sentences[-1] = updated_sentences[-1] + " " + statement
                else:
                    updated_sentences.append(statement)
            
            else:
                # cases where citations are included
                if "[" in statement or (s_index < len(sentences) - 1 and len(sentences[s_index+1]) > 0 and sentences[s_index+1][0] == "["):
                    updated_sentences.append(statement)
                else:
                    updated_sentences.append("[replace_{}]".format(s_index))
                    post_hoc_sentence["[replace_{}]".format(s_index)] = statement

        if len(post_hoc_sentence) > 0:
            print("{0} sentences require attributions, e..g, {1}".format(len(post_hoc_sentence), list(post_hoc_sentence.values())[0] ))
            prompts = []
            for s in list(post_hoc_sentence.values()):    
                input_query = instructions.posthoc_attributions_paragraph.format_map({"statement": s, "passages": passages})
                if llama3_chat is True:
                    input_query = create_prompt_with_llama3_format(input_query)
                
                prompts.append(input_query)
             
            outputs = []
            for input_query in prompts:
                raw_output = chat_completion(input_query, model=self.client_llm)
                outputs.append(raw_output)
            
            # Postprocess Output
            for output, sentence_key in zip(outputs, list(post_hoc_sentence.keys())):
                if len([t.split("[Response_End]")[0] for t in output.split("[Response_Start]") if "[Response_End]" in t]) == 0:
                    post_hoc_sentence[sentence_key] = post_hoc_sentence[sentence_key]
                else:
                    processed_output = [t.split("[Response_End]")[0] for t in output.split("[Response_Start]") if "[Response_End]" in t][0]
                    post_hoc_sentence[sentence_key] = processed_output
                
            final_processed_outputs = []
            for item in updated_sentences:
                if item in post_hoc_sentence:
                    final_processed_outputs.append(post_hoc_sentence[item])
                else:
                    final_processed_outputs.append(item)
            updated_sentences = final_processed_outputs
            
        return "\n".join(updated_sentences)
    
    def insert_attributions_posthoc(self, item, llama3_chat=False):
        text = item["output"]
        passages = item["final_passages"]

        sentences = sent_tokenize(text)
        # post process sentences 
        updated_sentences = []
        post_hoc_sentence = {}

        for s_index, statement in enumerate(sentences):
            if len(statement) < 10:
                if statement[0] == "[":
                    updated_sentences[-1]  = updated_sentences[-1] + " " + statement
                else:
                    updated_sentences.append(statement)
            
            else:
                # cases where citations are included
                if "[" in statement or (s_index < len(sentences) - 1 and sentences[s_index+1][0] =="["):
                    updated_sentences.append(statement)
                else:
                    updated_sentences.append("[replace_{}]".format(s_index))
                    post_hoc_sentence["[replace_{}]".format(s_index)] = statement

        if len(post_hoc_sentence) > 0:
                        
            print("{0} sentences require attributions, e..g, {1}".format(len(post_hoc_sentence), list(post_hoc_sentence.values())[0] ))
            prompts = []
            for s in list(post_hoc_sentence.values()):    
                input_query = instructions.posthoc_attributions.format_map({"statement": s, "passages": passages})

                if llama3_chat is True:
                    input_query = create_prompt_with_llama3_format(input_query)
                
                prompts.append(input_query)
            
            outputs = []
            for input_query in prompts:
                raw_output = chat_completion(input_query, model=self.client_llm)
                outputs.append(raw_output)
            
            # process_output
            for output, sentence_key in zip(outputs, list(post_hoc_sentence.keys())):
                if len([t.split("[Response_End]")[0] for t in output.split("[Response_Start]") if "[Response_End]" in t]) == 0:
                    post_hoc_sentence[sentence_key] = post_hoc_sentence[sentence_key]
                else:
                    processed_output = [t.split("[Response_End]")[0] for t in output.split("[Response_Start]") if "[Response_End]" in t][0]
                    post_hoc_sentence[sentence_key] = processed_output
                
            final_processed_outputs = []
            for item in updated_sentences:
                if item in post_hoc_sentence:
                    final_processed_outputs.append(post_hoc_sentence[item])
                else:
                    final_processed_outputs.append(item)
            updated_sentences = final_processed_outputs
            
        return " ".join(updated_sentences)

    def insert_attributions_posthoc_paragraph_all(self, item, llama3_chat=False):
        text = item["output"]
        if "final_passages" in item:
            passages = item["final_passages"] 
        else:
            ctxs = item["ctxs"]
            passages = ""
            for idx, p in enumerate(ctxs):
                passages += "[{0}] {1}\n".format(idx, p)

        sentences = text.split("\n")
        # print(sentences)
        updated_sentences = []
        post_hoc_sentence = {}
        prompts = []

        for s_index, statement in enumerate(sentences):
            if len(statement) < 10:
                if len(updated_sentences) > 0 and len(statement) > 0 and statement[0] == "[":
                    updated_sentences[-1] = updated_sentences[-1] + " " + statement
                else:
                    updated_sentences.append(statement)
            
            else:
                updated_sentences.append("[replace_{}]".format(s_index))
                post_hoc_sentence["[replace_{}]".format(s_index)] = statement

        for s in list(post_hoc_sentence.values()):    
            input_query = instructions.posthoc_attributions_paragraph_all.format_map({"statement": s, "passages": passages})

            if llama3_chat is True:
                input_query = create_prompt_with_llama3_format(input_query)
            
            prompts.append(input_query)
        
        outputs = []
        for input_query in prompts:
            raw_output = chat_completion(input_query, model=self.client_llm)
            outputs.append(raw_output)
        
        # process_output
        for output, sentence_key in zip(outputs, list(post_hoc_sentence.keys())):
            if len([t.split("[Response_End]")[0] for t in output.split("[Response_Start]") if "[Response_End]" in t]) == 0:
                post_hoc_sentence[sentence_key] = post_hoc_sentence[sentence_key]
            else:
                processed_output = [t.split("[Response_End]")[0] for t in output.split("[Response_Start]") if "[Response_End]" in t][0]
                post_hoc_sentence[sentence_key] = processed_output
            
        final_processed_outputs = []
        for item in updated_sentences:
            if item in post_hoc_sentence:
                final_processed_outputs.append(post_hoc_sentence[item])
            else:
                final_processed_outputs.append(item)
        updated_sentences = final_processed_outputs
        
        return "\n".join(updated_sentences)

    def run(self, item, ranking_ce=False, use_feedback=False, skip_generation=False, posthoc_at=False, llama3_chat=False, task_name="default", zero_shot=False, max_per_paper=None, use_abstract=False, max_tokens=3000):
        # print("llama3 chat format? {0}".format(llama3_chat), flush=True)
        # print("use feedback: {}".format(use_feedback), flush=True)
            
        if ranking_ce is True:
            item["ctxs"], ranked_results, id_mapping = self.reranking_passages_cross_encoder(item, use_abstract=False)
            item["ranked_results"] = ranked_results
            item["id_mapping"] = id_mapping

        if max_per_paper is not None:
            filtered_ctxs = []
            title_to_count = {}
            for ctx in item["ctxs"]:
                if "title" not in ctx or ctx["title"] is None:
                    ctx["title"] = ""
                title_to_count.setdefault(ctx["title"], 0)
                if title_to_count[ctx["title"]] > max_per_paper:
                    # print("We have already aded the paper {0} {1} times".format(ctx["title"], max_per_paper))
                    continue
                else:
                    filtered_ctxs.append(ctx)
                    title_to_count[ctx["title"]] += 1
                    
            item["ctxs"] = filtered_ctxs
            
        if skip_generation is False:
            generated_result, passages = self.generate_response(item, max_tokens=max_tokens, llama3_chat=llama3_chat, task_name=task_name, zero_shot=zero_shot)
            if "\n\n References":
                generated_result = generated_result.split("\n\n References")[0]
            item["initial_result"] = generated_result

        if use_feedback is True:
            # print("[OpenScholar.run] Generating feedback...", flush=True)
            feedbacks = self.get_feedback(item, llama3_chat=llama3_chat)[:3]

            item["feedbacks"] = feedbacks
            with tqdm(total=len(feedbacks[:3]), desc="Processing feedbacks") as pbar:
                for feedback_idx, feedback in enumerate(feedbacks[:3]):
                    pbar.update(1)
                    # currently only supports non retrieval feedback
                    if len(feedback[1]) == 0:
                        edited_answer = self.edit_with_feedback(item, feedback[0], llama3_chat=llama3_chat)
                        if "Here is the revised answer:\n\n" in edited_answer:
                            edited_answer = edited_answer.split("Here is the revised answer:\n\n")[1]

                        if len(item["output"]) > 0 and len(edited_answer) / len(item["output"]) > 0.9:
                            item["output"] = edited_answer
                            item["edited_answer_{}".format(feedback_idx)] = edited_answer
                        else:
                            print("skipping as edited answers got too short")
                    else:
                        new_papers = []
                        if self.ss_retriever is True:
                            new_keywords = self.retrieve_keywords(feedback[1])
                            paper_list = {}
                            if len(new_keywords) > 0:
                                # print(f"\n[Used Feedback] Relevant keywords: {new_keywords}", flush=True)
                                for keyword in new_keywords:  
                                    top_papers = search_paper_via_query(keyword)
                                    if top_papers is None:
                                        # print(f"Keywords: {keyword} | No pub related to the current keyword.", flush=True)
                                        pass
                                    else:
                                        for paper in top_papers:
                                            if paper["paperId"] not in paper_list:
                                                paper["text"] = paper["abstract"]
                                                paper["citation_counts"] = paper["citationCount"]
                                                paper_list[paper["paperId"]] = paper
                                new_papers += list(paper_list.values())
                                # remove duplicarted data 
                        if len(new_papers) > 0:
                            # print("before deduplication: {}".format(len(new_papers)))
                            new_papers_dicts = {paper["text"][:100] + paper["title"]: paper for paper in new_papers if paper is not None and type(paper["text"]) is str}
                            new_papers = list(new_papers_dicts.values())
                            # print("after deduplication: {}".format(len(new_papers)))
                            # add new papers when and only when we have the new papers. 
                            if len(new_papers) > 0:
                                new_passages_reranked, _ , _  = self.reranking_passages_cross_encoder_supplemental(item, new_papers)
                                passages_start_index = len(item["ctxs"])

                                edited_answer = self.edit_with_feedback_retrieval(item, feedback[0], new_passages_reranked, passages_start_index)

                                if len(item["output"]) > 0 and len(edited_answer) / len(item["output"]) > 0.9:
                                    item["ctxs"] += new_passages_reranked[:self.top_n]
                                    item["edited_answer_{}".format(feedback_idx)] = edited_answer
                                    item["output"] = edited_answer
                                    item["edited_answer_{}".format(feedback_idx)] = edited_answer
                                elif len(item["output"]) == 0 and len(edited_answer) > 0:
                                    item["ctxs"] += new_passages_reranked[:self.top_n]
                                    item["edited_answer_{}".format(feedback_idx)] = edited_answer
                                    item["output"] = edited_answer
                                    item["edited_answer_{}".format(feedback_idx)] = edited_answer
                                else:
                                    print("skipping as edited answers got too short")

        if posthoc_at is True:
            # attributed_results = self.insert_attributions_posthoc(item, llama3_chat=llama3_chat)
            # attributed_results = self.insert_attributions_posthoc_paragraph(item, llama3_chat=llama3_chat)
            attributed_results =  self.insert_attributions_posthoc_paragraph_all(item, llama3_chat=llama3_chat)
            item["output"] = attributed_results
        
        item["output"] = item["output"].replace("[Response_Start]", "").replace("[Response_End]", "")

        # print(item["output"])

        # if "\n### References" in item["output"]:
        #     item["output"] = item["output"].split("\n### References")[0]
        return item

def process_paragraph(text):
    text = text.replace("<cit.>", "")
    text = remove_citations(text)
    return text

def process_input_data(data, use_contexts=True):
    processed_data = []
    for item in data:
        if "answer" not in item:
            item["answer"] = ""
        if "input" not in item:
            if "question" in item:
                item["input"] = item["question"]
            if "query" in item:
                item["input"] = item["query"]

        new_ctxs = []
        if use_contexts is True:
            # normalize ctx format for different retrieval APIs
            for ctx in item["ctxs"]:
                if type(ctx) is list:
                    for c in ctx:
                        if type(c) is dict:
                            new_ctxs.append(c)
                if type(ctx) is dict:
                    new_ctxs.append(ctx)
            item["ctxs"] = new_ctxs

            # remove duplicated contexts
            processed_paras = []
            with tqdm(total=len(item["ctxs"])) as pbar:
                for ctx in item["ctxs"]:
                    pbar.update(1)
                    if "retrieval text" in ctx:
                        ctx["text"] = ctx["retrieval text"]
                    if ctx["text"] is None or len(ctx["text"]) ==0:
                        continue
                    if type(ctx["text"]) != str:
                        ctx["text"] = " ".join(ctx["text"]["contexts"])
                    ctx["text"] = process_paragraph(ctx["text"])
                    if "title" not in ctx:
                        ctx["title"] = ""
                    processed_paras.append(ctx)

            processed_paras_dicts = {paper["text"][:100] + paper["title"]: paper for paper in processed_paras}
            processed_paras = list(processed_paras_dicts.values())

            item["ctxs"] = processed_paras
            item["original_ctxs"] = processed_paras
        processed_data.append(item)
    return processed_data