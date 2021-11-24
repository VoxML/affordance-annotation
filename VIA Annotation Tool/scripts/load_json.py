import json

f = open('/s/red/a/nobackup/vision/anju/Downloads/hico-det-tool/data/hico_processed/anno_list.json',)
 
data = json.load(f)
 
f1 = open("/s/red/a/nobackup/vision/anju/Downloads/hico-det-tool/data/hico_processed/output/demofile3.txt", "w")
for annotation in data:
	f1.write("==============\n")
	f1.write(annotation["global_id"])
	for hois in annotation["hois"]:
		f1.write("\n")
		f1.write("---")
		f1.write("\n")
		f1.write(hois["id"])
		f1.write("\n")
		f1.write(str(hois["human_bboxes"]))
		f1.write("\n")
		f1.write(str(hois["object_bboxes"]))
		f1.write("\n")

f.close()
f1.close()


