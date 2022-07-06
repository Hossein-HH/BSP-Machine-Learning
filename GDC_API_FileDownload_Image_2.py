import requests
import json
import re
import os
import openslide as ops

def files_order(json_file):
    files = []
    with open(json_file,'r') as f:
        data = json.load(f)
        for d in data:
            files.append([d['file_name'],d['file_size']])
    files = sorted(files, key=lambda x: x[1])
    return files

def convert_to_png(filename):
    out_name = filename+'.png'
    image = ops.OpenSlide(filename)
    t_img = image.get_thumbnail(image.associated_images['thumbnail'].size)
    t_img.save(out_name)
    image.close()
    t_img.close()
    with open('download_files.txt', 'a') as f:
        f.write(filename)
        f.write('\n')
    # os.remove(filename)
    return out_name

if __name__ == "__main__":
    files_endpt = "https://api.gdc.cancer.gov/files"
    files = files_order("Clinical_data/BRCA/Image/files.2022-05-21.json")
    process_files = open('files.csv', 'a')
    for i,f in enumerate(files):
        if i<=195:
            continue
        print(f'File [{i+1}/{len(files)}]: size => {f[1]:,} bytes')
        name = f[0]
        filters = {
            "op": "and",
            "content":[
                {
                "op": "in",
                "content":{
                    "field": "cases.project.project_id",
                    "value": ["TCGA-BRCA"]
                    }
                },
                {
                "op": "in",
                "content":{
                    "field": "files.access",
                    "value": ["open"]
                    }
                },
                {
                "op": "in",
                "content":{
                    "field": "files.file_name",
                    "value": [name]
                    }
                },
                {
                "op": "in",
                "content":{
                    "field": "files.data_format",
                    "value": ["svs"]
                    }
                }
            ]
        }
        # Here a GET is used, so the filter parameters should be passed as a JSON string.
        params = {
            "filters": json.dumps(filters),
            "fields": "file_id",
            "format": "JSON",
            "size": "1"
            }
        response = requests.get(files_endpt, params = params)
        file_uuid_list = []
        # This step populates the download list with the file_ids from the previous query
        for file_entry in json.loads(response.content.decode("utf-8"))["data"]["hits"]:
            file_uuid_list.append(file_entry["file_id"])
        # print(file_uuid_list)
        # exit(0)
        data_endpt = "https://api.gdc.cancer.gov/data"
        params = {"ids": file_uuid_list}
        response = requests.post(data_endpt, data = json.dumps(params), headers = {"Content-Type": "application/json"})
        # print(response.content)
        # exit(0)
        response_head_cd = response.headers["Content-Disposition"]
        file_name = re.findall("filename=(.+)", response_head_cd)[0]
        # print(response_head_cd); exit(0)
        file_name = "Clinical_data/BRCA/Image/download/"+file_name
        print(file_name)
        with open(file_name, "wb") as output_file:
            output_file.write(response.content)
        converted_name = convert_to_png(filename=file_name)
        # print(f'\t{file_name}->{converted_name}')
        process_files.write(converted_name+'\n')
    process_files.close()