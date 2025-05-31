"""
This module defines unit tests for backend image-to-image generation module.

This module consists of 4 main types of tests:
1. Unit tests for individual functions and classes.
2. Consistency tests so that generated images are consistent with requests and source images.
3. Style tests to ensure that generator can copy art style and reuse it.
4. Diversity tests to ensure that generator can generate diverse images.
5. Performance tests to ensure that generator can generate images quickly.

Tests are evaluated by hand not by automated means because automated evaluation is not feasible due to the subjective nature of image quality and style.

testset.csv consists of these columns:
id: For each unit test to have a specific id defined by i2i-[id]
name: For each unit test to have an easily recognizable name
description: For each unit test to have a clear and concise description
input_groups: To know what inputs to use to generate output
tested_property: Describes how should the test be evaluated.

inputs.csv is a file that describes inputs and their groups
id: For each input to have a unique id
group: For each input to belong to a group
description: For each input to have a clear and concise description
path: To be findable in the project directory
"""
