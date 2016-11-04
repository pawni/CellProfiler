'''<b>Measure Neural Features</b> extracts features from the images using a pre-trained neural network.
'''
import cellprofiler.image as cpi
import cellprofiler.module as cpm
import cellprofiler.measurement as cpmeas
import cellprofiler.setting as cps
import tensorflow as tf
import numpy as np
import os.path as path
from cellprofiler.preferences import ABSOLUTE_FOLDER_NAME

class MeasureNeuralFeatures(cpm.Module):
    module_name = "MeasureNeuralFeatures"
    variable_revision_number = 1
    category = "Measurement"

    def create_settings(self):
        """Create the setting variables
        """
        self.image_name = cps.ImageNameSubscriber(
                "Select the input image", cps.NONE, doc='''
            Choose the image to be used to calculate the illumination function.''')

        self.batch_size = cps.Integer(
            "Batch Size", 256, 1, doc='''
                    Choose batch size for passing through network.''')

        self.directory = cps.DirectoryPath(
            "Location of model file",
            dir_choices=[
                ABSOLUTE_FOLDER_NAME], doc="""
                    Here you set the folder where the model is saved.""")
        self.directory.dir_choice = ABSOLUTE_FOLDER_NAME
        self.count = 0

    def settings(self):
        return [self.image_name, self.batch_size, self.directory]

    def visible_settings(self):
        """The settings as seen by the UI

        """
        result = [self.image_name, self.batch_size, self.directory]
        return result

    def help_settings(self):
        return [self.image_name, self.batch_size, self.directory]

    def prepare_run(self, workspace):
        return True

    def prepare_group(self, workspace, grouping, image_numbers):
        d = self.get_dictionary()['Neural'] = {}
        d['count'] = 0
        d['images'] = []
        d['image_set_numbers'] = []

        session = tf.Session()
        dir_path = self.directory.get_absolute_path()
        new_saver = tf.train.import_meta_graph(path.join(dir_path, 'model.meta'))
        new_saver.restore(session, path.join(dir_path, 'model'))
        graph = tf.get_default_graph()
        features = graph.get_tensor_by_name("features:0")
        input_placeholder = graph.get_tensor_by_name("input:0")

        d['session'] = session
        d['features'] = features
        d['input_placeholder'] = input_placeholder
        #TODO: include pipeline.run_group_with_yield
        return True

    def is_aggregation_module(self):
        return True

    def run(self, workspace):
        input_image_name = self.image_name.value

        image_set = workspace.image_set
        d = self.get_dictionary()['Neural']
        count = d['count']
        input_image = image_set.get_image(input_image_name,
                                          must_be_grayscale=True)
        meas = workspace.measurements
        assert isinstance(meas, cpmeas.Measurements)
        d['images'].append([input_image.pixel_data])
        d['image_set_numbers'].append(meas.image_set_number)

        if len(d['images']) >= self.batch_size.value:
            # push images through net and add measurements
            self.process_batch(workspace)

        d['count'] = count + 1

    def post_group(self, workspace, grouping):
        d = self.get_dictionary()['Neural']
        if len(d['images']) > 0:
            self.process_batch(workspace)
        # push images through net and add measurements

    def process_batch(self, workspace):
        input_image_name = self.image_name.value
        d = self.get_dictionary()['Neural']
        images = np.array(d['images'])
        input_batch = np.swapaxes(np.swapaxes(np.repeat(images, 3, axis=1), 1, 3), 1, 2)
        session = d['session']
        features = d['features']
        input_placeholder = d['input_placeholder']
        extracted_features = session.run((features), feed_dict={input_placeholder: input_batch})
        image_nums = d['image_set_numbers']
        meas = workspace.measurements
        assert isinstance(meas, cpmeas.Measurements)
        for i in xrange(extracted_features.shape[0]):
            for j in xrange(extracted_features.shape[1]):
                meas.add_measurement(cpmeas.IMAGE, 'Neural_%s_f%d' % (input_image_name, j), extracted_features[i, j],
                                     image_set_number=image_nums[i])
        d['images'] = []
        d['image_set_numbers'] = []

    def display(self, workspace, figure):
        # these are actually just the pixel data
        pass

    def get_measurement_columns(self, pipeline):
        input_image_name = self.image_name.value
        # TODO: change number of features to be dependent on model used
        return [(cpmeas.IMAGE,
                 'Neural_%s_f%d' % (input_image_name, feature), cpmeas.COLTYPE_FLOAT) for feature in xrange(2048)]

    #
    # get_categories returns a list of the measurement categories produced
    # by this module. It takes an object name - only return categories
    # if the name matches.
    #
    def get_categories(self, pipeline, object_name):
        if object_name == cpmeas.IMAGE:
            return ['Neural']
        else:
            # Don't forget to return SOMETHING! I do this all the time
            # and CP mysteriously bombs when you use ImageMath
            return []

    #
    # Return the feature names if the object_name and category match
    #
    def get_measurements(self, pipeline, object_name, category):
        input_image_name = self.image_name.value
        if (object_name == cpmeas.IMAGE and
                    category == 'Neural'):
            # TODO: change number of features to be dependent on model used
            return ['Neural_%s_f%d' % (input_image_name, feature) for feature in xrange(2048)]
        else:
            return []

    #
    # This module makes per-image measurements. That means we need
    # get_measurement_images to distinguish measurements made on two
    # different images by this module
    #
    def get_measurement_images(self, pipeline, object_name, category, measurement):
        #
        # This might seem wasteful, but UI code can be slow. Just see
        # if the measurement is in the list returned by get_measurements
        #
        if measurement in self.get_measurements(
                pipeline, object_name, category):
            return [self.image_name.value]
        else:
            return []
