import React, { useState } from 'react';
import { Dialog } from 'primereact/dialog';
import { FileUpload } from 'primereact/fileupload';
import { Button } from 'primereact/button';
import { exportPMSSettings } from '../../../api/api';
import LoadingSpinner from '../../LoadingSpinner/LoadingSpinner';


interface PMSSettingsDialogProps {
    visible: boolean;
    onClose: () => void;
}

const PMSSettingsDialog = ({ visible, onClose }: PMSSettingsDialogProps) => {

    // API call function
    const downloadFile = async () => {
        await exportPMSSettings();
    };

    return (
        <>
            <Dialog
                header="Download PMS settings"
                visible={visible}
                style={{ width: '20vw' }}
                onHide={onClose}
            >
                <div>
                    <p>
                        Download the PMS settings.
                    </p>
                    <div className="p-field">
                        <Button
                            type="button"
                            label="Download File"
                            icon="pi pi-download"
                            onClick={downloadFile}
                            className="p-button-success"
                        />
                    </div>
                </div>
            </Dialog>
        </>
    );
};

export default PMSSettingsDialog;