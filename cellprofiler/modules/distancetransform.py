"""

Distance transform

"""

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import scipy.ndimage
import skimage


class DistanceTransform(cellprofiler.module.Module):
    category = "Image processing"
    module_name = "Distance transform"
    variable_revision_number = 1

    def create_settings(self):
        self.x_name = cellprofiler.setting.ImageNameSubscriber(
            "Input"
        )

        self.y_name = cellprofiler.setting.ImageNameProvider(
            "Output",
            "distance"
        )

        self.distance = cellprofiler.setting.Choice(
            "Distance",
            [
                "Euclidean distance"
            ]
        )

    def settings(self):
        return [
            self.x_name,
            self.y_name,
            self.distance
        ]

    def visible_settings(self):
        return [
            self.x_name,
            self.y_name,
            self.distance
        ]

    def run(self, workspace):
        x_name = self.x_name.value
        y_name = self.y_name.value

        images = workspace.image_set

        x = images.get_image(x_name)

        x_data = x.pixel_data

        x_data = skimage.img_as_uint(x_data)

        y_data = scipy.ndimage.distance_transform_edt(x_data)

        y = cellprofiler.image.Image(
            image=y_data,
            parent_image=x
        )

        images.add(y_name, y)

        if self.show_window:
            workspace.display_data.x_data = x_data
            workspace.display_data.y_data = y_data

    def display(self, workspace, figure):
        dimensions = (2, 1)

        x_data = workspace.display_data.x_data[0]
        y_data = workspace.display_data.y_data[0]

        figure.set_subplots(dimensions)

        figure.subplot_imshow(
            0,
            0,
            x_data,
            colormap="gray"
        )

        figure.subplot_imshow(
            1,
            0,
            y_data,
            colormap="gray"
        )