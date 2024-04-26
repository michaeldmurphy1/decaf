import json

def print_json_structure(data, indent='', level=0, max_depth=100):
    """
    Recursively prints the structure of a JSON object with sample values.

    Parameters:
    - data: The JSON data to print the structure of.
    - indent: A string of spaces used to indent the printout for nested structures.
    - level: Current level of depth in the JSON structure.
    - max_depth: Maximum depth to print to avoid too verbose output.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                print(f"{indent}{key}: {type(value)}")
                if level < max_depth:
                    print_json_structure(value, indent + '  ', level+1, max_depth)
            else:
                print(f"{indent}{key}: {type(value)} [{str(value)[:50]}...]")
    elif isinstance(data, list):
        print(f"{indent}List of {len(data)} items: {type(data)}")
        # Optionally, explore the first item to get a sense of list contents
        if len(data) > 0 and level < max_depth:
            print(f"{indent}First item:")
            print_json_structure(data[0], indent + '  ', level+1, max_depth)

# Load the JSON file
json_file_path = 'analysis/data/MuonTrigSF/2018/2018_trigger/Efficiencies_muon_generalTracks_Z_Run2018_UL_SingleMuonTriggers_schemaV2.json'
with open(json_file_path, 'r') as file:
    json_data = json.load(file)

# Print the structure of the JSON data
#print("JSON Structure:")
#print_json_structure(json_data)

# print the keys of the json data
print("Keys of the JSON data:")


#print keys of the second key
#for i in print(json_data['corrections'][0]['data']['content'][i]):


for i in range(len(json_data['corrections'])):
    print(json_data['corrections'][i]['name'])
    #if json_data['corrections'][i]['name'] == 'NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight':
        #print(i)


print(json_data['corrections'][4]['name']['NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight']['content'])