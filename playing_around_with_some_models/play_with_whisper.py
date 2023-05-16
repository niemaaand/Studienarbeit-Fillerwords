import whisper

from Inspection.ModelInspection import get_model_size


def execute(filename):
    model = whisper.load_model("base")
    result = model.transcribe(filename, word_timestamps=True, initial_prompt="uh uhm a ... ", language="en")
    return result


def print_model_size():
    n_params, size = get_model_size(whisper.load_model('base'))
    print('Params: ' + str(n_params) + '\nModel size: {:.3f}MB'.format(size))
