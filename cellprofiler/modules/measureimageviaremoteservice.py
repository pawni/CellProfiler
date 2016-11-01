'''<b> MeasureImageViaRemoteService </b> allows you to send images from CellProfiler to a remote service, and record the measurements produced by that service.
<hr>
Remote image-processing services can accept incoming images from CellProfiler,
perform some image processing, and return to CellProfiler some measurement(s)
derived from the image. These services might exist because the processing
requires the provider's substantial computing resources, or because the
service provider wants to keep their source code proprietary, or
because the service provider wants to charge per-use.

This module allows you to specify the location of a particular
remote service and send an image there. When the provider
produces measurements, they are returned and stored in
CellProfiler as per-image measurements.

Authentication is assumed to happen outside of this module -
that is, you may need to log in to the remote service in
order for it to accept incoming images.

<h4>Available measurements</h4>
<ul>
<li><b>Image measurements:</b>
<ul> The measurements produced by this module are dynamically determined,
depending on what measurements the remote service provides to CellProfiler.
</ul>
</li>
'''
#################################
#
# Imports from useful Python libraries
#
#################################

import numpy as np
import scipy.ndimage as scind

#################################
#
# Imports from CellProfiler
#
# The package aliases are the standard ones we use
# throughout the code.
#
##################################

import cellprofiler.image as cpi
import cellprofiler.module as cpm
import cellprofiler.measurement as cpmeas
import cellprofiler.setting as cps


## custom imports
from grpc.beta import implementations
import inception_inference_pb2
import base64
import skimage
import scipy.misc as misc

##################################
#
# Constants
#
# I put constants that are used more than once here.
#
###################################

'''This is the measurement template category'''
C_REMOTE_SERVICE = "RemoteService"


###################################
#
# The module class
#
# Your module should "inherit" from cellprofiler.cpmodule.CPModule.
# This means that your module will use the methods from CPModule unless
# you re-implement them. You can let CPModule do most of the work and
# implement only what you need.
#
###################################

class MeasureImageViaRemoteService(cpm.Module):
    ###############################################
    #
    # The module starts by declaring the name that's used for display,
    # the category under which it is stored and the variable revision
    # number which can be used to provide backwards compatibility if
    # you add user-interface functionality later.
    #
    ###############################################
    module_name = "MeasureImageViaRemoteService"
    category = "Measurement"
    variable_revision_number = 1

    ###############################################
    #
    # create_settings is where you declare the user interface elements
    # (the "settings") which the user will use to customize your module.
    #
    # You can look at other modules and in cellprofiler.settings for
    # settings you can use.
    #
    ################################################

    def create_settings(self):
        #
        # The ImageNameSubscriber "subscribes" to all ImageNameProviders in
        # prior modules. Modules before yours will put images into CellProfiler.
        # The ImageSubscriber gives your user a list of these images
        # which can then be used as inputs in your module.
        #
        self.input_image_name = cps.ImageNameSubscriber(
                # The text to the left of the edit box
                "Input image name:",
                # HTML help that gets displayed when the user presses the
                # help button to the right of the edit box
                doc="""Select the image you want to send to the remote service.""")

        #
        #
        #
        #
        self.service_location = cps.Text(
            "Remote service (URL or IP)", "localhost", doc="""Provide the location (URL or IP address) of the service to use."""
        )
        self.service_port = cps.Integer(
                "Port of the remote service (integer)", 9000, minval=1,
                doc="""Provide the port of the service to use.""")

    #
    # The "settings" method tells CellProfiler about the settings you
    # have in your module. CellProfiler uses the list for saving
    # and restoring values for your module when it saves or loads a
    # pipeline file.
    #
    # This module does not have a "visible_settings" method. CellProfiler
    # will use "settings" to make the list of user-interface elements
    # that let the user configure the module. See imagetemplate.py for
    # a template for visible_settings that you can cut and paste here.
    #
    def settings(self):
        return [self.input_image_name, self.service_location,
                self.service_port]

    #
    # CellProfiler calls "run" on each image set in your pipeline.
    # This is where you do the real work.
    #
    def run(self, workspace):
        print('opening channel at %s and port %d' % (self.service_location.value, self.service_port.value))
        self.channel = implementations.insecure_channel(self.service_location.value,
                                                        int(self.service_port.value))

        self.stub = inception_inference_pb2.beta_create_InceptionService_stub(self.channel)
        #
        # Get the measurements object - we put the measurements we
        # make in here
        #
        meas = workspace.measurements
        assert isinstance(meas, cpmeas.Measurements)
        #
        # We record some statistics which we will display later.
        # We format them so that Matplotlib can display them in a table.
        # The first row is a header that tells what the fields are.
        #
        statistics = [["Class", "Score"]]
        #
        # Put the statistics in the workspace display data so we
        # can get at them when we display
        #
        workspace.display_data.statistics = statistics
        #
        # Get the input image and object. You need to get the .value
        # because otherwise you'll get the setting object instead of
        # the string name.
        #
        input_image_name = self.input_image_name.value
        ################################################################
        #
        # GETTING AN IMAGE FROM THE IMAGE SET
        #
        # Get the image set. The image set has all of the images in it.
        #
        image_set = workspace.image_set
        #
        # Get the input image object. We want a grayscale image here.
        # The image set will convert a color image to a grayscale one
        # and warn the user.
        #
        input_image = image_set.get_image(input_image_name,
                                          must_be_grayscale=True)

        #
        # Get the pixels - these are a 2-d Numpy array.
        #
        pixels = skimage.color.gray2rgb(input_image.pixel_data)
        print(pixels.shape)
        disp_lines, scores, classes = self.get_service_response(pixels)
        print('got lines:\n%s' % disp_lines)

        # statistics.append(disp_lines)
        # statistics.append([scores, classes])
        for i, row in enumerate(zip(classes, scores)):
            statistics.append(row)
            meas.add_image_measurement('class%d' % (i + 1), row[0])
            meas.add_image_measurement('score%d' % (i + 1), row[1])

        # #
        # #
        # # Then ask for the minimum_enclosing_circle for each object named
        # # in those indexes. MEC returns the i and j coordinate of the center
        # # and the radius of the circle and that defines the circle entirely.
        # #
        # centers, radius = minimum_enclosing_circle(labels, indexes)
        # ###############################################################
        # #
        # # The module computes a measurement based on the image intensity
        # # inside an object times a Zernike polynomial inscribed in the
        # # minimum enclosing circle around the object. The details are
        # # in the "measure_zernike" function. We call into the function with
        # # an N and M which describe the polynomial.
        # #
        # for n, m in self.get_zernike_indexes():
        #     # Compute the zernikes for each object, returned in an array
        #     zr, zi = self.measure_zernike(
        #             pixels, labels, indexes, centers, radius, n, m)
        #     # Get the name of the measurement feature for this zernike
        #     feature = self.get_measurement_name(n, m)
        #     # Add a measurement for this kind of object
        #     if m != 0:
        #         meas.add_measurement(input_object_name, feature, zr)
        #         #
        #         # Do the same with -m
        #         #
        #         feature = self.get_measurement_name(n, -m)
        #         meas.add_measurement(input_object_name, feature, zi)
        #     else:
        #         # For zero, the total is the sum of real and imaginary parts
        #         meas.add_measurement(input_object_name, feature, zr + zi)
        #     #
        #     # Record the statistics.
        #     #
        #     zmean = np.mean(zr)
        #     zmedian = np.median(zr)
        #     zsd = np.std(zr)
        #     statistics.append([feature, zmean, zmedian, zsd])


    def get_service_response(self, pixels):
        print('trying to access stub')
        # data = base64.b64encode(pixels)
        request = inception_inference_pb2.InceptionRequest()
        misc.imsave('image.jpg', pixels)
        with open('image.jpg', 'rb') as f:
            data = f.read()
            request.jpeg_encoded = data

        #request.jpeg_encoded = data
        # request.data = data
        result = str(self.stub.Classify(request, 30.00))
        print('got result: %s' % result)
        classes = []
        scores = []
        lines = []
        for line in result.split('\n'):
            if len(line.strip()) == 0:
                continue
            print('processing line %s' % line)
            (line_type, line_val) = line.split(':')
            line_val = line_val.strip()
            if line_type.startswith('score'):
                scores.append(float(line_val))
                lines.append(float(line_val))
            elif line_type.startswith('class'):
                classes.append(line_val[1:-1])
                lines.append(line_val[1:-1])
        disp_lines = []
        for i in range(len(lines) / 2):
            disp_lines.append([lines[(len(lines) / 2) + i], lines[i]])
        return disp_lines, scores, classes

    ################################
    #
    # DISPLAY
    #
    def display(self, workspace, figure=None):
        statistics = workspace.display_data.statistics
        if figure is None:
            figure = workspace.create_or_find_figure(subplots=(1, 1,))
        else:
            figure.set_subplots((1, 1))
        figure.subplot_table(0, 0, statistics)

    #######################################
    #
    # Here, we go about naming the measurements.
    #
    # Measurement names have parts to them, traditionally separated
    # by underbars. There's always a category and a feature name
    # and sometimes there are modifiers such as the image that
    # was measured or the scale at which it was measured.
    #
    # We have functions that build the names so that we can
    # use the same functions in different places.
    #

    #
    # We have to tell CellProfiler about the measurements we produce.
    # There are two parts: one that is for database-type modules and one
    # that is for the UI. The first part gives a comprehensive list
    # of measurement columns produced. The second is more informal and
    # tells CellProfiler how to categorize its measurements.
    #
    #
    # get_measurement_columns gets the measurements for use in the database
    # or in a spreadsheet. Some modules need the pipeline because they
    # might make measurements of measurements and need those names.
    #
    def get_measurement_columns(self, pipeline):
        #
        # We use a list comprehension here.
        # See http://docs.python.org/tutorial/datastructures.html#list-comprehensions
        # for how this works.
        #
        # The first thing in the list is the object being measured. If it's
        # the whole image, use cpmeas.IMAGE as the name.
        #
        # The second thing is the measurement name.
        #
        # The third thing is the column type. See the COLTYPE constants
        # in measurement.py for what you can use
        #
        return [(cpmeas.IMAGE,
                 'class%d' % (i / 2 + 1) if i % 2 == 0 else 'score%d' % (i / 2 + 1),
                 cpmeas.COLTYPE_VARCHAR if i % 2 == 0 else cpmeas.COLTYPE_FLOAT)
                for i in range(10)]

    #
    # get_categories returns a list of the measurement categories produced
    # by this module. It takes an object name - only return categories
    # if the name matches.
    #
    def get_categories(self, pipeline, object_name):
        if object_name == cpmeas.IMAGE:
            return [C_REMOTE_SERVICE]
        else:
            # Don't forget to return SOMETHING! I do this all the time
            # and CP mysteriously bombs when you use ImageMath
            return []

    #
    # Return the feature names if the object_name and category match
    #
    def get_measurements(self, pipeline, object_name, category):
        if (object_name == cpmeas.IMAGE and
                    category == C_REMOTE_SERVICE):
            return ["Magic"]
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
            return [self.input_image_name.value]
        else:
            return []