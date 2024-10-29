import numpy as np
def gen_meta_MnM2(vendor, scanners, disease, field, size):   ## This is for MnM2 MetaData
    temp = np.zeros(size)
    # Mapping vendors to numerical values
    vendor_mapping = {'Philips Medical Systems': 1, 'SIEMENS': 2, 'GE MEDICAL SYSTEMS': 3}
    temp[0] = vendor_mapping.get(vendor, 0)
    # Mapping scanners to numerical values
    scanners_mapping = {
        'Symphony': 4, 'SIGNA EXCITE': 3, 'Signa Explorer': 2,
        'SymphonyTim': 1, 'Avanto Fit':0 , 'Avanto': -1,
        'Achieva': -2, 'Signa HDxt': -3, 'TrioTim': -4
    }
    temp[1] = scanners_mapping.get(scanners, 0)
    # Mapping diseases to numerical values
    disease_mapping = {'NOR': 3, 'LV': 2, 'HCM': 1, 'ARR': 0, 'FALL': -1, 'CIA': -2}
    temp[2] = disease_mapping.get(disease, 0)
    # Mapping field to numerical values
    temp[3] = float(field)
    return temp
vendor_gt = {
    'GE MEDICAL SYSTEMS': 0,
    'SIEMENS': 1,
    'Philips Medical Systems': 2,
}
scanner_gt = {
        'Symphony': 0, 'SIGNA EXCITE': 1, 'Signa Explorer': 2,
        'SymphonyTim': 3, 'Avanto Fit': 4, 'Avanto': 5,
        'Achieva': 6, 'Signa HDxt': 7, 'TrioTim': 8
    }
disease_gt = {'NOR': 0, 'LV': 1, 'HCM': 2, 'ARR': 3, 'FALL': 4, 'CIA': 5}
field_gt = {'1.5': 0, '3.0': 1}


sex_mapping = {'M': 0, 'F': 1}
quality_mapping = {'Good': 0, 'Medium': 1, 'Poor': 2}
def gen_meta_CAMUS(sex, image_quality,es,nb_frame,age,ef,frame_rate, size):   ## This is for CAMUS MetaData
    temp = np.zeros(size,dtype=float)
    temp[0] = float(sex_mapping[sex])
    temp[1] = float(quality_mapping[image_quality])
    temp[2] = float(es)
    temp[3] = float(nb_frame)
    temp[4] = float(age)
    temp[5] = float(ef)
    temp[6] = float(frame_rate)
    return temp
