from pipeline.dicom import dicom_to_nifti, organize_dicom_files


class Dicom(object):

    def convert(self, input_directory, output_directory, biobank):
        dicom_id, biobank_id = dicom_to_nifti(input_directory, output_directory, biobank)
        return dicom_id, biobank_id

    def organize(self, input_directory, output_directory, biobank):
        _, _ = organize_dicom_files(input_directory, output_directory, biobank)
