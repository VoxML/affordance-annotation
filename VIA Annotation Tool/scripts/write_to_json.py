import json

beginning="""{
  "_via_settings": {
    "ui": {
      "annotation_editor_height": 25,
      "annotation_editor_fontsize": 0.8,
      "leftsidebar_width": 18,
      "image_grid": {
        "img_height": 80,
        "rshape_fill": "none",
        "rshape_fill_opacity": 0.3,
        "rshape_stroke": "yellow",
        "rshape_stroke_width": 2,
        "show_region_shape": true,
        "show_image_policy": "all"
      },
      "image": {
        "region_label": "__via_region_id__",
        "region_color": "__via_default_region_color__",
        "region_label_font": "10px Sans",
        "on_image_annotation_editor_placement": "NEAR_REGION"
      }
    },
    "core": {
      "buffer_size": 18,
      "filepath": {
        
      },
      "default_filepath": ""
    },
    "project": {
      "name": "via234"
    }
  },
  "_via_img_metadata": {
    
  },
  "_via_attributes": {
    "region": {
      "action": {
        "type": "dropdown",
        "description": "",
        "options": {"""


f1 = open('/s/red/a/nobackup/vision/anju/Downloads/hico-det-tool/data/hico_processed/verb_list.json',)
data = json.load(f1)
temp=""
for annotation in data:
	temp+="\n"+'"'+annotation["name"]+'": "",'
verbs = temp.rstrip(temp[-1])
verbs+="\n"+'},'+"\n"
#options="\n"+'"walk": "",' +"\n" +'"bike": ""'+"\n"+'},'+"\n"
ending=""""default_options": {
          
        }
      }
    },
    "file": {
      
    }
  },
  "_via_data_format_version": "2.0.10",
  "_via_image_id_list": [
    
  ]
}"""
f2 = open("/s/red/a/nobackup/vision/anju/Downloads/hico-det-tool/data/hico_processed/output/data.json", "w")

for line in beginning:
	f2.write(line)
f2.write(verbs)
f2.write(ending)
f2.close()
f1.close()
"""
"walk": "",
          "bike": ""
        },
        "default_options": {
          
        }
      }
    },
    "file": {
      
    }
  },
  "_via_data_format_version": "2.0.10",
  "_via_image_id_list": [
    
  ]
}'
"""


