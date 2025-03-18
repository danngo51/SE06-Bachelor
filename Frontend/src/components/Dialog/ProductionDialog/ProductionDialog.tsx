import React, { useState } from 'react';
import { Dialog } from 'primereact/dialog';
import { Dropdown } from 'primereact/dropdown';
import { Button } from 'primereact/button';
import { FileUpload } from 'primereact/fileupload';
import styles from './ProductionDialog.module.css';
import { ProfileType } from '../../../data/profileConfRequestDTO';
import { updateProfile } from '../../../api/api';
import { ProfileConfRequest } from '../../../data/profileConfRequestDTO';
import LoadingSpinner from '../../LoadingSpinner/LoadingSpinner';

interface ProductionProfilesDialogProps {
    visible: boolean;
    onClose: () => void;
}

const ProductionDialog = ({
    visible,
    onClose,
}: ProductionProfilesDialogProps) => {
    const [formData, setFormData] = useState<ProfileConfRequest>({
        excelFile: null!, // Initially null, but File is guaranteed when submitting
        profileType: ProfileType.DEFAULT,
        shouldOverwriteTables: false,
        specificCountriesToUpdate: '',
    });

    const [loading, setLoading] = useState(false);

    // Options for dropdowns
    const profileTypeOptions = [
        { label: 'Select a profile type', value: null }, // Placeholder
        { label: 'Default', value: ProfileType.DEFAULT },
        { label: 'CHP', value: ProfileType.CHP },
        { label: 'Hydro', value: ProfileType.HYDRO },
        { label: 'Solar', value: ProfileType.SOLAR },
        { label: 'Nuclear', value: ProfileType.NUCLEAR },
        { label: 'Wind onshore', value: ProfileType.WIND_ONSHORE },
        { label: 'Wind offshore', value: ProfileType.WIND_OFFSHORE },
    ];

    const isSubmitDisabled =
        !formData.excelFile ||
        formData.profileType == null ||
        !formData.specificCountriesToUpdate ||
        !formData.shouldOverwriteTables;

    // Handle form submission
    const handleSubmit = () => {
        const requestData = new FormData();

        if (
            !formData.excelFile ||
            formData.profileType == null ||
            !formData.specificCountriesToUpdate ||
            !formData.shouldOverwriteTables
        ) {
            return;
        }

        setLoading(true);

        // Append the fields to the FormData object
        if (formData.profileType) {
            requestData.append('profileType', formData.profileType.toString());
        }
        if (formData.excelFile) {
            requestData.append('excelFile', formData.excelFile);
        }
        requestData.append(
            'shouldOverwriteTables',
            String(formData.shouldOverwriteTables)
        );
        requestData.append(
            'specificCountriesToUpdate',
            formData.specificCountriesToUpdate
        );

        updateProfile(requestData)
            .then(() => {
                onClose();
            })
            .catch((error) => {
                console.error('Failed to update profile:', error);
            });

        setLoading(false);

        onClose(); // Close the dialog
    };

    // Handle file selection using PrimeReact FileUpload component
    const handleFileUpload = (event: any) => {
        const uploadedFile = event.files[0]; // Grab the first file from the uploaded files
        setFormData({ ...formData, excelFile: uploadedFile });
    };

    return (
        <>
            <LoadingSpinner loading={loading} />
            <Dialog
                header="Indlæs produktionsprofiler"
                visible={visible}
                style={{ width: '30vw' }}
                onHide={onClose}
            >
                <form
                    onSubmit={(e) => {
                        e.preventDefault();
                        handleSubmit();
                    }}
                >
                    <div className={`p-fluid ${styles.dialogContainer}`}>
                        {/* Profile type input */}
                        <div className="p-field">
                            <label htmlFor="dropdown1">Profil type</label>
                            <Dropdown
                                className={`${styles.dialogElement}`}
                                id="dropdown1"
                                value={formData.profileType}
                                options={profileTypeOptions}
                                onChange={(e) =>
                                    setFormData({
                                        ...formData,
                                        profileType: e.target
                                            .value as ProfileType,
                                    })
                                }
                                placeholder="Select an option"
                            />
                        </div>

                        {/* Overwrite table input */}
                        <div>
                            <label>
                                <input
                                    type="checkbox"
                                    checked={formData.shouldOverwriteTables}
                                    onChange={(e) =>
                                        setFormData({
                                            ...formData,
                                            shouldOverwriteTables:
                                                e.target.checked,
                                        })
                                    }
                                />
                                Overwrite Tables
                            </label>
                        </div>

                        {/* Specific countries input */}
                        <div>
                            <label>Specific Countries to Update:</label>
                            <input
                                className={`${styles.customInput}`}
                                type="text"
                                placeholder="Comma-separated countries"
                                onChange={(e) =>
                                    setFormData({
                                        ...formData,
                                        specificCountriesToUpdate:
                                            e.target.value,
                                    })
                                }
                            />
                        </div>

                        {/* File upload using PrimeReact FileUpload */}
                        <div className="p-field">
                            <label htmlFor="fileUpload">File Upload</label>
                            <FileUpload
                                mode="basic"
                                id="fileUpload"
                                name="excelFile"
                                accept=".xlsx, .xls"
                                // maxFileSize={1000000}  // Optional: Set a max file size (1MB example)
                                onUpload={handleFileUpload} // Handle file upload
                                chooseLabel="Choose Excel File"
                                cancelLabel="Cancel"
                            />
                        </div>

                        <div className="p-field">
                            <Button
                                type="submit"
                                label="Submit"
                                icon="pi pi-check"
                                disabled={isSubmitDisabled}
                                className="p-button-success"
                            />
                        </div>
                    </div>
                </form>
            </Dialog>
        </>
    );
};

export default ProductionDialog;
