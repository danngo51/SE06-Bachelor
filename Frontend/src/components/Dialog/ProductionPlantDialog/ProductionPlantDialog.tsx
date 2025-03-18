import React, { useState } from 'react';
import { Dialog } from 'primereact/dialog';
import { FileUpload } from 'primereact/fileupload';
import { Button } from 'primereact/button';
import styles from './ProductionPlantDialog.module.css';
import { uploadExcelFile } from '../../../api/api';
import { deletePowerPlants } from '../../../api/api';
import LoadingSpinner from '../../LoadingSpinner/LoadingSpinner';

interface ProductionProfilesDialogProps {
    visible: boolean;
    onClose: () => void;
}

const ProductionPlantDialog = ({
    visible,
    onClose,
}: ProductionProfilesDialogProps) => {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [loading, setLoading] = useState(false);

    // Handle file selection using 'onSelect'
    const handleFileSelect = (event: any) => {
        const uploadedFile = event.files[0]; // Grab the first selected file
        setSelectedFile(uploadedFile);
    };

    // API call function
    const callUploadExcelAPI = async () => {
        if (!selectedFile) {
            console.error('No file selected.');
            return;
        }
        setLoading(true);

        // Call your API functions
        await deletePowerPlants();
        await uploadExcelFile(selectedFile);

        setLoading(false);
    };

    return (
        <>
            <LoadingSpinner loading={loading} />

            <Dialog
                header="Upload excel file with production plants"
                visible={visible}
                style={{ width: '30vw' }}
                onHide={onClose}
            >
                <div className={`p-fluid ${styles.dialogContainer}`}>
                    {/* File Upload */}
                    <div>
                        <FileUpload
                            className={`${styles.dialogElement}`}
                            mode="basic"
                            id="fileUpload"
                            name="file"
                            accept=".xlsx, .xls"
                            chooseLabel="Choose file"
                            onSelect={handleFileSelect} // Correct event to capture the file
                            auto={false} // Prevent auto-upload
                        />
                    </div>
                    {/* Submit Button */}
                    <div className="p-field">
                        <Button
                            label="Upload File"
                            icon="pi pi-upload"
                            className={`p-button-success ${styles.dialogElement}`}
                            onClick={callUploadExcelAPI}
                            disabled={!selectedFile} // Disable if no file is selected
                        />
                    </div>
                </div>
            </Dialog>
        </>
    );
};

export default ProductionPlantDialog;
