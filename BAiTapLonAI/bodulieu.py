from torch.utils.data import Dataset
from xulyaudio import AudioUtil

# ----------------------------
# Sound Dataset
# ----------------------------

class SoundDS(Dataset):
    def __init__(self, df,data_path):
        self.df=df
        self.data_path = str(data_path)
        self.duration = 6000
        self.sr = 44100
        self.channel = 2
        self.shift_pct = 0.4
            
    # ----------------------------
    # Number of items in dataset
    # ----------------------------
    def __len__(self):
        return len(self.df)    
    
    # ----------------------------
    # Get i'th item in dataset
    # ----------------------------
    def __getitem__(self,idx):
        # Absolute file path of the audio file - concatenate the audio directory with
        # the relative path
        audio_file = self.data_path + self.df.loc[idx, 'relative_path']
        sentence = self.df.loc[idx, 'sentence']
        # Get the Class ID
        unique_words = list(set(" ".join(sentence).split()))
        word_to_id = {word: idx for idx, word in enumerate(unique_words)}
        # Tạo từ điển ánh xạ từ ID sang từ
        id_to_word = {idx: word for idx, word in enumerate(unique_words)}

# Ánh xạ các từ trong mỗi nhãn sang ID
        labels_as_ids = [[word_to_id[word] for word in label.split()] for label in sentence]
        aud = AudioUtil.open(audio_file)
        # Some sounds have a higher sample rate, or fewer channels compared to the
        # majority. So make all sounds have the same number of channels and same 
        # sample rate. Unless the sample rate is the same, the pad_trunc will still
        # result in arrays of different lengths, even though the sound duration is
        # the same.
        reaud = AudioUtil.resample(aud, self.sr)
        rechan = AudioUtil.rechannel(reaud, self.channel)

        dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
        shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
        
        
        
        
        mfcc=AudioUtil.createMFCC(shift_aud)
        aug_sgram = AudioUtil.spectro_augment(mfcc, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
        mfcc_slices=AudioUtil.split_mfcc(aug_sgram,1000)
        return mfcc_slices, labels_as_ids