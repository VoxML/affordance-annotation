import json
import ast
import collections
import numpy as np
from ordered_set import OrderedSet


'''

f_1000_items = open("C:\\Users\\anjug\\CodeWorkspace\\Research\\hicodet_annotations\\annotations\\Jan_8\\via234_1000_items_jan8.json",)

data = json.load(f_1000_items)

f_region_attribs = open("C:\\Users\\anjug\\CodeWorkspace\\Research\\hicodet_annotations\\output\\region_attribs.txt", "w")


img_metadata=data["_via_img_metadata"]


for key in img_metadata:
    f_region_attribs.write(key)
    f_region_attribs.write("\n")
    regions=img_metadata[key]['regions']
    for entry in regions:
        f_region_attribs.write(str(entry['region_attributes']))
        f_region_attribs.write("\n")
    f_region_attribs.write("\n-------------------\n")

f_region_attribs.write("\n")

f_1000_items.close()
f_region_attribs.close()

'''

#f_region_attribs = open(r"C:\Users\anjug\CodeWorkspace\Research\hicodet_annotations\output\region_attribs_small.txt", 'r')
f_region_attribs = open(r"C:\Users\anjug\CodeWorkspace\Research\hicodet_annotations\output\region_attribs.txt", 'r')

Lines = f_region_attribs.readlines()
#f_matrix = open(r"C:\Users\anjug\CodeWorkspace\Research\hicodet_annotations\output\matrix.txt", "w")

actions_global_dict_up={}
unique_objects_orderedSet=OrderedSet()
unique_actions_orderedSet=OrderedSet()
Lines=iter(Lines)
for line in Lines:
    #print("line :44 ",line)
    if line.startswith("HICO"):
        #print(line)
        nextLine=next(Lines)
        #print("line :48 ",nextLine)
        dict_ob={}
        dict_hum={}
        i=1;
        while(not(nextLine.startswith("---"))):
            #print("nextLine ** ",nextLine)
            if(nextLine.startswith('{')):
                #print("i: ",i)
                if("category': 'object" in nextLine):
                    dict_ob[i]=nextLine
                else:
                    dict_hum[i]=nextLine
                nextLine=next(Lines)
                if(not(nextLine.startswith(('{')))):
                    nextLine=next(Lines)
                #print("line :59 ",nextLine)
                i=i+1
        #nextLine=next(Lines)
        #print("line :62 ",nextLine)
        object_id_list=[]
        
        for key,value in dict_hum.items():
            print("\n\n\n")
            print(" &&&& KEY is : ",key)
            print(" &&&& VALUE is : ",value)
            res=str(value).replace("'", '"')
            
            dictionary = ast.literal_eval(res)
           
            actions_local_list=dictionary['action'].split(',')
            unique_actions_orderedSet.update(actions_local_list)
            
            #------------------------# Test Case - If field 'obj id' is null -------------------------------------------------
            o=dictionary['obj id']
            if len(o)==0: #If field 'obj id' is null
                break;
            #------------------------- End of Test Case ----------------------------------------------------------------------
            
            object_id_list=dictionary['obj id'].split(',')
            
            
            #------------------------# Test Case - If field 'obj id' has non-numeric components-------------------------------
            object_id_list_isdigit = [s for s in object_id_list if s.isdigit()]
            
            if(len(object_id_list_isdigit)!=len(object_id_list)):
                break;
            #------------------------- End of Test Case ----------------------------------------------------------------------
           
            
            #------------------------# Test Case - If the obj id in human annotations does not exist in the list of obj ids---
            is_obj_id_not_exists = False
            for x in object_id_list:
                
                if(int(x) not in dict_ob.keys()):
                    is_obj_id_not_exists = True
            if is_obj_id_not_exists:
                break;
            #--------------------------- End of Test Case ---------------------------------------------------------------------
            
            #----------- # Test Case - If the size of the object_id_list does not match the size of the actions_local_list-----
            
            if(len(object_id_list) != len(actions_local_list)):
                continue
            
            #--------------------------- End of Test Case ---------------------------------------------------------------------
         
            
            for entry in actions_local_list:
                if actions_global_dict_up.get(entry) is None:
                    actions_global_dict_up[entry]=[]
            
           
            for i in range(len(object_id_list)):
                print("\n\n")
                print("** new i **")
                value=dict_ob.get(int(object_id_list[i]))
                print("value : ",value)
                res=str(value).replace("'", '"')
                dictionary = ast.literal_eval(res)
                
                obj_name = dictionary['obj name']
                unique_objects_orderedSet.add(obj_name)
                
                
                up = dictionary['up']
                print("i : ",i)
                print("object_id_list : ",object_id_list)
                print("object_id_list[i] : ",object_id_list[i])
                print("actions_local_list : ",actions_local_list)
                print("actions_local_list[i] : ",actions_local_list[i])
                action=actions_local_list[i]
                
                for key,value1 in actions_global_dict_up.items():
                    
                    is_duplicate=False
                    matching_obj_name=""
                    for dict in value1:
                        if(obj_name in dict.keys()):
                            is_duplicate=True
                            matching_obj_name=obj_name
                    
                    if(action==key):
                        plus_x=1 if "+x" in up else 0
                        minus_x=1 if "-x" in up else 0
                        plus_y=1 if "+y" in up else 0
                        minus_y=1 if "-y" in up else 0
                        plus_z=1 if "+z" in up else 0
                        minus_z=1 if "-z" in up else 0
                        n_a=1 if "n/a" in up else 0
                        
                        if(not is_duplicate):
                           dict1 = {obj_name: {"+x":plus_x,"-x":minus_x,"+y":plus_y,"-y":minus_y,"+z":plus_z,"-z":minus_z,"n/a":n_a}}
                           actions_global_dict_up[action].append(dict1)    
                        else:
                            for dict in value1:
                                if(obj_name in dict.keys()):
                                    dict[obj_name]["+x"]+=plus_x
                                    dict[obj_name]["-x"]+=minus_x
                                    dict[obj_name]["+y"]+=plus_y
                                    dict[obj_name]["-y"]+=minus_y
                                    dict[obj_name]["+z"]+=plus_z
                                    dict[obj_name]["-z"]+=minus_z
                                    dict[obj_name]["n/a"]+=n_a

actions_up_array=np.zeros((len(unique_actions_orderedSet),len(unique_objects_orderedSet),7))#(depth,rows,columns)

depth=-1
row=-1
for key,value in actions_global_dict_up.items():
    depth+=1
    for dict1 in value:       
        for key1,value1 in dict1.items():
            row=-1
            for entry in unique_objects_orderedSet:
                row+=1
                if(key1==entry):
                    actions_up_array[depth][row][0]=value1["+x"]
                    actions_up_array[depth][row][1]=value1["-x"]
                    actions_up_array[depth][row][2]=value1["+y"]
                    actions_up_array[depth][row][3]=value1["-y"]
                    actions_up_array[depth][row][4]=value1["+z"]
                    actions_up_array[depth][row][5]=value1["-z"]
                    actions_up_array[depth][row][6]=value1["n/a"]
                    
         
#print("*****actions_up_array*******")
#print(actions_up_array)

print("-------------------------------------------------------------")
print("***** actions_global_dict_up *******")
print("-------------------------------------------------------------")
print(actions_global_dict_up)


print("-------------------------------------------------------------")
print("***** Objects and Actions OrderedSets *******")
print("-------------------------------------------------------------")
print(unique_objects_orderedSet)
print(unique_actions_orderedSet)


actions_up_cond_prob=np.zeros((len(unique_actions_orderedSet),len(unique_objects_orderedSet),7))#(depth,rows,columns)
actions_up_joint_prob=np.zeros((len(unique_actions_orderedSet),len(unique_objects_orderedSet),7))#(depth,rows,columns)


#Calculate Conditional Probability
''' Calculate P(+x/object,action) '''

#Calculate Joint Probability
''' Calculate P(+x,object,action) '''

total_sum_array=actions_up_array.sum()
count_joint_prob= total_sum_array if total_sum_array!=0 else 1
for i in range(len(actions_up_array)):
    for j in range(len(actions_up_array[i])):
        sum_row=sum(actions_up_array[i][j])
        count_cond_prob= sum_row if sum_row!=0 else 1
        for k in range(len(actions_up_array[i][j])):
            cond_prob  = actions_up_array[i][j][k]/count_cond_prob
            joint_prob = actions_up_array[i][j][k]/count_joint_prob
            actions_up_cond_prob[i][j][k]  = cond_prob
            actions_up_joint_prob[i][j][k] = joint_prob

f_actions_up_array = open("actions_up_array.txt", "a")
f_actions_up_cond_prob = open("actions_up_cond_prob.txt", "a")
f_actions_up_joint_prob = open("actions_up_joint_prob.txt", "a")


print("-------------------------------------------------------------")            
print("***** actions_up_array *******")
print("-------------------------------------------------------------")
#print(actions_up_array)
f_actions_up_array.write(str(actions_up_array))

print("-------------------------------------------------------------")
print("***** Calculate Conditional Probabilities *******")
print("-------------------------------------------------------------")
print("-------------------------------------------------------------")
print("***** actions_up_cond_prob array *******")
print("-------------------------------------------------------------")
#print(actions_up_cond_prob)
f_actions_up_cond_prob.write(str(actions_up_cond_prob))


print("-------------------------------------------------------------")
print("***** Calculate Joint Probabilities *******")
print("-------------------------------------------------------------")
print("-------------------------------------------------------------")
print("*****actions_up_joint_prob array*******")
print("-------------------------------------------------------------")
print("Total Sum of elements of array is : ",total_sum_array)
#print(actions_up_joint_prob)
f_actions_up_joint_prob.write(str(actions_up_joint_prob))

f_actions_up_array.close()
f_actions_up_cond_prob.close()
f_actions_up_joint_prob.close()