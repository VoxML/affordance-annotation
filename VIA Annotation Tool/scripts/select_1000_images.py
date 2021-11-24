import json 

f1 = open('/s/red/a/nobackup/vision/anju/Downloads/hico-det-tool/data/hico_processed/hoi_list.json',)
data1=json.load(f1)
Dict_hoi_object={}
for annotation in data1:
	hoi_id=annotation["id"]
	object_name=annotation["object"]
	Dict_hoi_object[hoi_id]=object_name
f1.close()

'''
002  apple 
010  bicycle
014  bottle      
019  car
023  chair
027  cup
029  dog
038  horse
042  knife 
077  umbrella
'''

output_locn = "/s/red/a/nobackup/vision/anju/Downloads/hico-det-tool/data/hico_processed/output/"
apple_list=output_locn+"apple.txt"
bicycle_list=output_locn+"bicycle.txt"
bottle_list=output_locn+'bottle.txt'
car_list=output_locn+"car.txt"
chair_list=output_locn+"chair.txt"
cup_list=output_locn+"cup.txt"
dog_list=output_locn+"dog.txt"
horse_list=output_locn+"horse.txt"
knife_list=output_locn+"knife.txt"
umbrella_list=output_locn+"umbrella.txt"

f_apple=open(apple_list,"w")
f_bicycle=open(bicycle_list,"w")
f_bottle=open(bottle_list,"w")
f_car=open(car_list,"w")
f_chair=open(chair_list,"w")
f_cup=open(cup_list,"w")
f_dog=open(dog_list,"w")
f_horse=open(horse_list,"w")
f_knife=open(knife_list,"w")
f_umbrella=open(umbrella_list,"w")

f2 = open('/s/red/a/nobackup/vision/anju/Downloads/hico-det-tool/data/hico_processed/anno_list.json',)
data2 = json.load(f2)
for annotation in data2:
	img_name=annotation["global_id"]+".jpg"
	for hois in annotation["hois"]:
		hoi=hois["id"]
		if(Dict_hoi_object[hoi]=="apple"):
			f_apple.write(img_name)
			f_apple.write("\n")
		if(Dict_hoi_object[hoi]=="bicycle"):
			f_bicycle.write(img_name)
			f_bicycle.write("\n")
		if(Dict_hoi_object[hoi]=="bottle"):
			f_bottle.write(img_name)
			f_bottle.write("\n")
		if(Dict_hoi_object[hoi]=="car"):
			f_car.write(img_name)
			f_car.write("\n")
		if(Dict_hoi_object[hoi]=="chair"):
			f_chair.write(img_name)
			f_chair.write("\n")
		if(Dict_hoi_object[hoi]=="cup"):
			f_cup.write(img_name)
			f_cup.write("\n")
		if(Dict_hoi_object[hoi]=="dog"):
			f_dog.write(img_name)
			f_dog.write("\n")
		if(Dict_hoi_object[hoi]=="horse"):
			f_horse.write(img_name)
			f_horse.write("\n")
		if(Dict_hoi_object[hoi]=="knife"):
			f_knife.write(img_name)
			f_knife.write("\n")
		if(Dict_hoi_object[hoi]=="umbrella"):
			f_umbrella.write(img_name)
			f_umbrella.write("\n")

f_apple.close()
f_bicycle.close()
f_bottle.close()
f_car.close()
f_chair.close()
f_cup.close()
f_dog.close()
f_horse.close()
f_knife.close()
f_umbrella.close()

