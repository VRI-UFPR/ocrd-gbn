import os.path

import ocrd_modelfactory
import ocrd_models.ocrd_page as ocrd_page
import ocrd_models.ocrd_page_generateds as ocrd_page_gends
import ocrd_utils

from gbn.processor import OcrdGbnProcessor
from gbn.lib.dl import Model
from gbn.lib.struct import Contour
from gbn.lib.util import pil_to_cv2_rgb


class OcrdGbnSbbCrop(OcrdGbnProcessor):
    tool = "ocrd-gbn-sbb-crop"
    log = ocrd_utils.getLogger("processor.OcrdGbnSbbCrop")

    def process(self):
        # Ensure path to model is absolute:
        self.parameter['model'] = os.path.realpath(self.parameter['model'])

        # Construct Model object:
        model = Model(self.parameter['model'], self.parameter['shaping'])

        for (self.page_num, self.input_file) in enumerate(self.input_files):
            self.log.info(
                "Processing input file: %i / %s",
                self.page_num,
                self.input_file
            )

            # Create a new PAGE file from the input file:
            page_id = self.input_file.pageId or self.input_file.ID
            pcgts = ocrd_modelfactory.page_from_file(
                self.workspace.download_file(self.input_file)
            )
            page = pcgts.get_Page()

            # Get image from PAGE:
            page_image, page_xywh, _ = self.workspace.image_from_page(
                page,
                page_id,
                feature_selector=self.parameter['feature_selector'],
                feature_filter=self.parameter['feature_filter']
            )

            # Convert PIL to cv2 (RGB):
            page_image, _ = pil_to_cv2_rgb(page_image)

            # Get prediction for segment:
            page_prediction = model.predict(page_image)

            # Find contours of prediction:
            contours = Contour.from_image(page_prediction.img)

            # Filter out child contours:
            contours = list(filter(lambda cnt: not cnt.is_child(), contours))

            # Filter out invalid polygons:
            contours = list(
                filter(lambda cnt: cnt.polygon.is_valid(), contours)
            )

            # Sort contours by area:
            contours = sorted(contours, key=lambda cnt: cnt.area)

            # Get polygon of largest contour:
            border_polygon = contours[-1].polygon

            self._set_Border(
                page,
                page_image,
                page_xywh,
                border_polygon.points
            )

            # Add metadata about this operation:
            metadata = pcgts.get_Metadata()
            metadata.add_MetadataItem(
                ocrd_page_gends.MetadataItemType(
                    type_="processingStep",
                    name=self.ocrd_tool['steps'][0],
                    value=self.tool,
                    Labels=[
                        ocrd_page_gends.LabelsType(
                            externalModel="ocrd-tool",
                            externalId="parameters",
                            Label=[
                                ocrd_page_gends.LabelType(
                                    type_=name,
                                    value=self.parameter[name]
                                ) for name in self.parameter.keys()
                            ]
                        )
                    ]
                )
            )

            # Save XML PAGE:
            self.workspace.add_file(
                ID=self.page_file_id,
                file_grp=self.page_grp,
                pageId=page_id,
                mimetype=ocrd_utils.MIMETYPE_PAGE,
                local_filename=os.path.join(
                    self.output_file_grp,
                    self.page_file_id
                    ) + ".xml",
                content=ocrd_page.to_xml(pcgts)
            )
