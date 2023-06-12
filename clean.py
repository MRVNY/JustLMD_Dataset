from GLOBAL import *

for year_dir in songs_collection:
        for song in os.listdir(year_dir):
            song_path = year_dir + song
            # remove the folder annots 
            os.system("rm -rf " + song_path + "/annots")
            os.system("rm -rf " + song_path + "/audios")
            os.system("rm -rf " + song_path + "/images")
            os.system("rm -rf " + song_path + "/videos")
            os.system("mv " + song_path + "/output-smpl-3d/smplfull.json " + song_path + "/smplfull.json")
            os.system("rm -rf " + song_path + "/output-smpl-3d")
            os.system("rm " + song_path + "/*.yml")
            os.system("rm " + song_path + "/*.mp4")
            os.system("rm " + song_path + "/*.mp3")