import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import speech_recognition as sr
import io
from pydub import AudioSegment

tokenizer = Wav2Vec2Processor.from_pretrained('indonesian-nlp/wav2vec2-large-xlsr-indonesian')
model = Wav2Vec2ForCTC.from_pretrained('indonesian-nlp/wav2vec2-large-xlsr-indonesian')
r = sr.Recognizer()

with sr.Microphone(sample_rate=16000) as source:
  print ('coba ngomong')
  while True:
    audio = r.listen(source)
    data = io.BytesIO(audio.get_wav_data())
    clip = AudioSegment.from_wav(data)
    x = torch.FloatTensor(clip.get_array_of_samples())

    inputs = tokenizer(x, sampling_rate=16000, return_tensors = 'pt', padding = 'longest').input_values
    logits = model(inputs).logits
    tokens = torch.argmax(logits, axis=-1)
    text = tokenizer.batch_decode(tokens)

    print ('Kalimat yang disebutkan: ', str(text).lower())