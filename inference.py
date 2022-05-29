from PIL import Image
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype


class TritonImageNetFeatureExtractorModel:
    def __init__(self, triton_host, model_name, model_version='1'):
        self.triton_client = httpclient.InferenceServerClient(url=triton_host)

        # configure input/output
        model_metadata = self.triton_client.get_model_metadata(
            model_name=model_name, model_version=model_version
        )

        model_input_metadata = model_metadata['inputs'][0]

        self.model_input_name = model_input_metadata['name']
        self.model_input_datatype = model_input_metadata['datatype']

        model_output_metadata = model_metadata['outputs'][0]

        self.model_output_name = model_output_metadata['name']

        self.model_name = model_name
        self.model_version = model_version

    def preprocessing(self, image, triton_datatype='FP32'):
        # 1) resize image to target height
        img = image.resize((224, 224), Image.Resampling.BILINEAR)

        # 2) convert to tensor (numpy)
        img_np_tensor = np.array(img)

        # 3) change order of channels
        img_np_tensor = np.transpose(img_np_tensor, (2, 0, 1))

        # 4) normalization
        img_np_tensor = img_np_tensor / 255.0
        mean = np.asarray((0.485, 0.456, 0.406)).reshape((-1, 1, 1))
        std = np.asarray((0.229, 0.224, 0.225)).reshape((-1, 1, 1))
        img_np_tensor = (img_np_tensor - mean) / std

        # 5) to triton type
        npdtype = triton_to_np_dtype(triton_datatype)
        typed = img_np_tensor.astype(npdtype)

        return typed

    def inference(self, image_tensor):
        t_input = httpclient.InferInput(self.model_input_name, image_tensor.shape, self.model_input_datatype)
        t_input.set_data_from_numpy(image_tensor)

        t_output = httpclient.InferRequestedOutput(self.model_output_name)

        response = self.triton_client.infer(
            self.model_name,
            [t_input],
            request_id='0',
            model_version=self.model_version,
            outputs=[t_output]
        )

        output_array = response.as_numpy(self.model_output_name)
        #
        # # post-process labels
        # score, label_num, label = [], [], []
        # for result in output_array:
        #     rs = result.decode('utf-8').split(':')
        #
        #     score.append(float(rs[0]))
        #     label_num.append(rs[1])
        #     label.append(rs[2])
        #
        # # softmax(score), forget to add it to the model
        # score = np.exp(score) / sum(np.exp(score))
        #
        # output_i = np.argmax(score)
        #
        # score = score[output_i]
        # label_num = label_num[output_i]
        # label = label[output_i]

        return output_array

