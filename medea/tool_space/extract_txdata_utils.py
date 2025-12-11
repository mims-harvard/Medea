import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List
from urllib import parse, request
from urllib.parse import urlparse, parse_qs, urlencode
import requests, re, time, mygene, zlib
from requests.adapters import HTTPAdapter, Retry
from xml.etree import ElementTree


LEGACY_UNIPROT_API_URL = 'https://legacy.uniprot.org/uploadlists/'
UNIPROT_API_URL = 'https://rest.uniprot.org/idmapping/'
OT_URL = "https://api.platform.opentargets.org/api/v4/graphql"
TOTAL_MAX = 20000
QUERY_BATCH_SIZE = 2048
POLLING_INTERVAL = 3


def get_disease_descendants(disease: str, source: str = 'ot', curated_disease_dir: str = None):
    """
    Get all descendants of a disease.
    """
    if source == 'ot':
        # Get all descendants of disease from OT
        flag = 0
        for fn in os.listdir(curated_disease_dir):
            with open(curated_disease_dir + fn) as f:
                diseases = f.readlines()
                for dis in diseases:
                    dis = json.loads(dis)
                    if dis['id'] == disease:
                        flag = 1
                        try:
                            all_disease = dis['descendants'] + [disease]
                            print(f'[pinnacle]: {disease} has {len(all_disease)-1} descendants')
                        except:
                            print(f'[pinnacle]: found {disease} has no descendants')
                            all_disease = [disease]
                            break
        assert flag == 1, f'{disease} not found in current database!'
    
    elif source == 'efo':
        # Get all descendants of that disease directly from EFO

        if disease.split('_')[0] == 'MONDO':
            efo_hierdesc = 'https://www.ebi.ac.uk/ols/api/ontologies/efo/terms/http%253A%252F%252Fpurl.obolibrary.org%252Fobo%252F' + disease + '/hierarchicalDescendants?size=5000'
        elif disease.split('_')[0] == 'EFO':
            efo_hierdesc = 'https://www.ebi.ac.uk/ols/api/ontologies/efo/terms/http%253A%252F%252Fwww.ebi.ac.uk%252Fefo%252F' + disease + '/hierarchicalDescendants?size=5000'
        elif disease.split('_')[0] == 'Orphanet':
            efo_hierdesc = 'https://www.ebi.ac.uk/ols/api/ontologies/orphanet/terms/http%253A%252F%252Fwww.orpha.net%252FORDO%252F' + disease + '/hierarchicalDescendants?size=5000'
        else:
            raise NotImplementedError

        disease_descendants = requests.request('GET', efo_hierdesc)
        assert disease_descendants.status_code==200

        # First, read the disease files and curate all diseases in this therapeutic area.
        raw_disease = json.loads(disease_descendants.text)
        assert raw_disease['page']['totalPages']==1

        all_disease = [disease]
        for raw in raw_disease['_embedded']['terms']:
            all_disease.append(raw['short_form'])
            assert raw['short_form'].split('_') == raw['obo_id'].split(':')
            try:
                for id in raw['annotation']['database_cross_reference']:
                    all_disease.append(id.replace(':', '_'))
            except:
                pass
            
        all_disease = set(all_disease)
    
    return all_disease


def get_all_drug_evidence(evidence_files: List, evidence_dir: str, all_disease: List, chembl2db: dict):
    """
    Get all target-disease associations with clinically relevant evidence, i.e. mediated by approved drugs / clinical candidate >= II (must be 'Completed' if II)
    """
    all_evidence = []
    for file in evidence_files:
        evidence_file = evidence_dir + file
        with open(evidence_file) as f:
            raw_evidence = f.readlines()
            evidence_list = [json.loads(evidence) for evidence in raw_evidence]

        for evidence in evidence_list:
            if ('diseaseFromSourceMappedId' in evidence.keys()) and ('clinicalPhase' in evidence.keys()) and (evidence['diseaseFromSourceMappedId'] in all_disease) and ((evidence['clinicalPhase']>=3) or (evidence['clinicalPhase']==2 and 'clinicalStatus' in evidence.keys() and evidence['clinicalStatus']=='Completed')):
                if 'clinicalStatus' in evidence.keys():
                    try:
                        all_evidence.append([evidence['diseaseFromSourceMappedId'], evidence['diseaseId'], evidence['targetId'], evidence['targetFromSourceId'], evidence['clinicalPhase'], evidence['clinicalStatus'], chembl2db[evidence['drugId']]])
                    except:
                        all_evidence.append([evidence['diseaseFromSourceMappedId'], evidence['diseaseId'], evidence['targetId'], evidence['targetFromSourceId'], evidence['clinicalPhase'], evidence['clinicalStatus'], evidence['drugId']])
                else:
                    try:
                        all_evidence.append([evidence['diseaseFromSourceMappedId'], evidence['diseaseId'], evidence['targetId'], evidence['targetFromSourceId'], evidence['clinicalPhase'], np.nan, chembl2db[evidence['drugId']]])
                    except: 
                        all_evidence.append([evidence['diseaseFromSourceMappedId'], evidence['diseaseId'], evidence['targetId'], evidence['targetFromSourceId'], evidence['clinicalPhase'], np.nan, evidence['drugId']])
                        
            elif ('diseaseId' in evidence.keys()) and ('clinicalPhase' in evidence.keys()) and (evidence['diseaseId'] in all_disease) and ((evidence['clinicalPhase']>=3) or (evidence['clinicalPhase']==2 and 'clinicalStatus' in evidence.keys() and evidence['clinicalStatus']=='Completed')):
                if 'clinicalStatus' in evidence.keys():
                    try:    
                        all_evidence.append([evidence['diseaseFromSourceMappedId'], evidence['diseaseId'], evidence['targetId'], evidence['targetFromSourceId'], evidence['clinicalPhase'], evidence['clinicalStatus'], chembl2db[evidence['drugId']]])
                    except:
                        all_evidence.append([evidence['diseaseFromSourceMappedId'], evidence['diseaseId'], evidence['targetId'], evidence['targetFromSourceId'], evidence['clinicalPhase'], evidence['clinicalStatus'], evidence['drugId']])
                else:
                    try:
                        all_evidence.append([evidence['diseaseFromSourceMappedId'], evidence['diseaseId'], evidence['targetId'], evidence['targetFromSourceId'], evidence['clinicalPhase'], np.nan, chembl2db[evidence['drugId']]])
                    except: 
                        all_evidence.append([evidence['diseaseFromSourceMappedId'], evidence['diseaseId'], evidence['targetId'], evidence['targetFromSourceId'], evidence['clinicalPhase'], np.nan, evidence['drugId']])
            
    drug_evidence_data = pd.DataFrame(all_evidence, columns=['diseaseFromSourceMappedId', 'diseaseId', 'targetId', 'targetFromSourceId', 'clinicalPhase', 'clinicalStatus', 'drugId']).sort_values(by='targetId')  # actually, it's drug-mediated target-disease association evidence data
    assert drug_evidence_data.diseaseFromSourceMappedId.isin(all_disease).all()
    assert drug_evidence_data.clinicalPhase.isin([2,3,4]).all()

    return drug_evidence_data


def get_all_associated_targets(disease: str):
    """
    Get all target-disease associations, except for those with only text mining (literature) evidence.
    """
    # Get all kinds of valid drug-disease associations
    def try_get_targets(index: int, size: int, all_targets: list, disease_id: str, query_string: str):
        """
        Try get targets for a disease from the API for the region of indices that contains the stale index.
        """
        if size!=1:
            index_temp = index * 2
            size_temp = size // 2
            for idx in [index_temp, index_temp+1]:
                variables = {"efoId":disease_id, "index":idx, 'size':size_temp}
                r = requests.post(OT_URL, json={"query": query_string, "variables": variables})
                assert r.status_code == 200
                try:
                    api_response = json.loads(r.text)
                    if type(api_response['data']['disease']['associatedTargets']['rows'])==list:
                        all_targets.extend(api_response['data']['disease']['associatedTargets']['rows'])
                    else:
                        all_targets.append(api_response['data']['disease']['associatedTargets']['rows'])
                except:
                    print(f"The stale index is within {str(idx*size_temp)}~{str((idx+1)*size_temp-1)}")
                    try_get_targets(idx, size_temp, all_targets, disease_id, query_string)
        else:
            print(f"Found stale index at index: {str(index)}!")
        
        return

    query_string = """
        query disease($efoId: String!, $index: Int!, $size: Int!) {
        disease(efoId: $efoId){
            id
            name
            associatedTargets(page: { index: $index, size: $size }) {
            rows {
                score
                datatypeScores{
                    id
                    score
                }
                target {
                    id
                    approvedSymbol
                }
            }
            }
        }
        }
    """

    all_targets = []
    for index in range(TOTAL_MAX//QUERY_BATCH_SIZE + 1):
        # Set variables object of arguments to be passed to endpoint
        variables = {"efoId":disease, "index":index, 'size':QUERY_BATCH_SIZE}

        # Perform POST request and check status code of response
        r = requests.post(OT_URL, json={"query": query_string, "variables": variables})
        assert r.status_code == 200

        # Transform API response from JSON into Python dictionary and print in console
        try:
            api_response = json.loads(r.text)
            all_targets.extend(api_response['data']['disease']['associatedTargets']['rows'])
        except:
            print(f'Unknown error when quering OT for {disease}.  Attemtping to get around the stale record...')
            try_get_targets(index, QUERY_BATCH_SIZE, all_targets, disease, query_string)

    all_associated_targets = [tar['target']['approvedSymbol'] for tar in all_targets if (len(tar['datatypeScores'])>1 or tar['datatypeScores'][0]['id']!='literature')]  #  All proteins associated with the disease, excluding those with only text mining evidence
    ensg2otgenename = {tar['target']['id']:tar['target']['approvedSymbol'] for tar in all_targets}

    print(f'[pinnacle]: Found {len(all_associated_targets)} associated targets for {disease}.')

    return all_associated_targets, ensg2otgenename


def evidence2genename(drug_evidence_data: pd.DataFrame, ensg2otgenename: dict):
    """
    Convert ENSG id and UniProt id in evidence to gene name through three ways combined, i.e. get all targets with clinically relevant evidence before intersecting with cell type PPIs
    """
    # UniProt
    try:
        uniprot_list = ' '.join(drug_evidence_data.targetFromSourceId.unique())
        params = {
            'from': 'ACC+ID',
            'to': 'GENENAME',
            'format': 'tab',
            'query': uniprot_list
        }

        data = parse.urlencode(params)
        data = data.encode('utf-8')
        req = request.Request(LEGACY_UNIPROT_API_URL, data)
        with request.urlopen(req) as f:
            response = f.read()
        res = response.decode('utf-8')
        uniprot2name = {ins.split('\t')[0]:ins.split('\t')[1] for ins in res.split('\n')[1:-1]}

    except:
        # Adapted from https://www.uniprot.org/help/id_mapping
        retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
        session = requests.Session()
        session.mount("https://", HTTPAdapter(max_retries=retries))

        def submit_job(src, dst, ids):
            """
            Submit job to UniProt ID mapping server, where `ids` is a str of identifiers separated by ','.
            """
            r = requests.post(
                f"{UNIPROT_API_URL}/run", 
                data={"from": src, "to": dst, "ids": ids},
            )
            r.raise_for_status()
            return r.json()["jobId"]

        def get_next_link(headers):
            re_next_link = re.compile(r'<(.+)>; rel="next"')
            if "Link" in headers:
                match = re_next_link.match(headers["Link"])
                if match:
                    return match.group(1)

        def check_id_mapping_results_ready(job_id):
            while True:
                r = session.get(f"{UNIPROT_API_URL}/status/{job_id}")
                r.raise_for_status()
                job = r.json()
                if "jobStatus" in job:
                    if job["jobStatus"] == "RUNNING":
                        print(f"Retrying in {POLLING_INTERVAL}s")
                        time.sleep(POLLING_INTERVAL)
                    else:
                        raise Exception(job["jobStatus"])
                else:
                    return bool(job["results"] or job["failedIds"])

        def get_batch(batch_response, file_format, compressed):
            batch_url = get_next_link(batch_response.headers)
            while batch_url:
                batch_response = session.get(batch_url)
                batch_response.raise_for_status()
                yield decode_results(batch_response, file_format, compressed)
                batch_url = get_next_link(batch_response.headers)

        def combine_batches(all_results, batch_results, file_format):
            if file_format == "json":
                for key in ("results", "failedIds"):
                    if key in batch_results and batch_results[key]:
                        all_results[key] += batch_results[key]
            elif file_format == "tsv":
                return all_results + batch_results[1:]
            else:
                return all_results + batch_results
            return all_results

        def get_id_mapping_results_link(job_id):
            url = f"{UNIPROT_API_URL}/details/{job_id}"
            request = session.get(url)
            request.raise_for_status()
            return request.json()["redirectURL"]

        def decode_results(response, file_format, compressed):
            if compressed:
                decompressed = zlib.decompress(response.content, 16 + zlib.MAX_WBITS)
                if file_format == "json":
                    j = json.loads(decompressed.decode("utf-8"))
                    return j
                elif file_format == "tsv":
                    return [line for line in decompressed.decode("utf-8").split("\n") if line]
                elif file_format == "xlsx":
                    return [decompressed]
                elif file_format == "xml":
                    return [decompressed.decode("utf-8")]
                else:
                    return decompressed.decode("utf-8")
            elif file_format == "json":
                return response.json()
            elif file_format == "tsv":
                return [line for line in response.text.split("\n") if line]
            elif file_format == "xlsx":
                return [response.content]
            elif file_format == "xml":
                return [response.text]
            return response.text

        def get_xml_namespace(element):
            m = re.match(r"\{(.*)\}", element.tag)
            return m.groups()[0] if m else ""

        def merge_xml_results(xml_results):
            merged_root = ElementTree.fromstring(xml_results[0])
            for result in xml_results[1:]:
                root = ElementTree.fromstring(result)
                for child in root.findall("{http://uniprot.org/uniprot}entry"):
                    merged_root.insert(-1, child)
            ElementTree.register_namespace("", get_xml_namespace(merged_root[0]))
            return ElementTree.tostring(merged_root, encoding="utf-8", xml_declaration=True)

        def print_progress_batches(batch_index, size, total):
            n_fetched = min((batch_index + 1) * size, total)
            print(f"[pinnacle]: Fetched evidence: {n_fetched} / {total}")

        def get_id_mapping_results_search(url):
            parsed = urlparse(url)
            query = parse_qs(parsed.query)
            file_format = query["format"][0] if "format" in query else "json"
            if "size" in query:
                size = int(query["size"][0])
            else:
                size = 500
                query["size"] = size
            compressed = (
                query["compressed"][0].lower() == "true" if "compressed" in query else False
            )
            parsed = parsed._replace(query=urlencode(query, doseq=True))
            url = parsed.geturl()
            request = session.get(url)
            request.raise_for_status()
            results = decode_results(request, file_format, compressed)
            total = int(request.headers["x-total-results"])
            print_progress_batches(0, size, total)
            for i, batch in enumerate(get_batch(request, file_format, compressed), 1):
                results = combine_batches(results, batch, file_format)
                print_progress_batches(i, size, total)
            if file_format == "xml":
                return merge_xml_results(results)
            return results

        def get_id_mapping_results_stream(url):
            if "/stream/" not in url:
                url = url.replace("/results/", "/stream/")
            request = session.get(url)
            request.raise_for_status()
            parsed = urlparse(url)
            query = parse_qs(parsed.query)
            file_format = query["format"][0] if "format" in query else "json"
            compressed = (
                query["compressed"][0].lower() == "true" if "compressed" in query else False
            )
            return decode_results(request, file_format, compressed)
        
        job_id = submit_job(
            src="UniProtKB_AC-ID", 
            dst="Gene_Name", 
            ids=drug_evidence_data.targetFromSourceId.unique().tolist()
        )

        if check_id_mapping_results_ready(job_id):
            link = get_id_mapping_results_link(job_id)
            results = get_id_mapping_results_search(link)

        uniprot2name = {rec['from']:rec['to'] for rec in results['results']}

    # ENSG --> gene name through mygene
    mg = mygene.MyGeneInfo()
    out = mg.querymany(drug_evidence_data.targetId.unique())
    ensg2name = {}
    for o in out:
        ensg2name[o['query']] = o['symbol']

    # Not sure why these didn't get added
    if "ENSG00000187733" not in ensg2otgenename: ensg2otgenename["ENSG00000187733"] = "AMY1C"
    if "ENSG00000231021" not in ensg2otgenename: ensg2otgenename["ENSG00000231021"] = "HLA-DRB4"
    if "ENSG00000276011" not in ensg2otgenename: ensg2otgenename["ENSG00000276011"] = "KIR2DL2"
    
    disease_drug_targets = set(uniprot2name.values())
    disease_drug_targets.update(ensg2name.values())
    # print(ensg2otgenename)
    # ENSG --> gene name through OT
    disease_drug_targets.update([ensg2otgenename[ensg] for ensg in drug_evidence_data.targetId])

    print(f'[pinnacle]: Found {len(disease_drug_targets)} targets with clinically relevant evidence.')
    
    return disease_drug_targets
