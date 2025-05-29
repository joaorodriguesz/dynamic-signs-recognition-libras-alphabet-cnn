import os

class VideoManager:
    def __init__(self, base_directory):
        self.base_directory = base_directory
    
    def list_videos(self, directory):
        """Lista todos os arquivos de vídeo (.mp4) em um diretório."""
        video_path = os.path.join(self.base_directory, directory)
        return [os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith(".mp4")]

    def count_videos(self, directory):
        """Conta quantos vídeos existem no diretório."""
        return len(self.list_videos(directory))
    
    def organize_videos_by_letter(self):
        """Cria um dicionário organizando os vídeos por letra do alfabeto."""
        video_dict = {}
        for dataset in os.listdir(self.base_directory):
            dataset_path = os.path.join(self.base_directory, dataset)
            if os.path.isdir(dataset_path):
                video_dict[dataset] = {}
                for letter in os.listdir(dataset_path):
                    letter_path = os.path.join(dataset_path, letter)
                    if os.path.isdir(letter_path):
                        video_dict[dataset][letter] = self.list_videos(f"{dataset}/{letter}")
        return video_dict
