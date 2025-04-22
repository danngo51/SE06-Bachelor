import React, { useState, useEffect } from 'react';
import { Dialog } from 'primereact/dialog';
import styles from './PredictDialog.module.css';
import { areaConnectionDTO } from '../../../data/transmissionConfRequestDTO';
import LoadingSpinner from '../../LoadingSpinner/LoadingSpinner';

import { FileUpload } from 'primereact/fileupload';

import { getAreaConnections } from '../../../api/api';
import { uploadAreaConnectionsFile } from '../../../api/api';
import { postAreaConnection } from '../../../api/api';

interface PredictDialogProps {
    visible: boolean;
    onClose: () => void;
}

const PredictDialog = ({ visible, onClose }: PredictDialogProps) => {
    const [areaConnection, setAreaConnection] = useState<areaConnectionDTO>({
        areaA: 0,
        areaB: 0,
        atoBCapacity: 0,
        btoACapacity: 0,
        connectionName: '',
        startYear: 0,
        stopYear: 0,
    });
    const [selectedFile, setSelectedFile] = useState<File | null>(null);

    const [formVisible, setFormVisible] = useState(false);
    const [loading, setLoading] = useState(false);
    const [dataList, setDataList] = useState<areaConnectionDTO[]>([]); // State for DataTable data

    const isDisabled =
        !areaConnection.areaA ||
        areaConnection.areaB == null ||
        !areaConnection.atoBCapacity ||
        !areaConnection.btoACapacity ||
        !areaConnection.connectionName ||
        !areaConnection.startYear ||
        !areaConnection.stopYear ||
        areaConnection.stopYear != areaConnection.startYear;

    //Get area connections
    useEffect(() => {
        // Fetch data from API
        fetchData();
    }, []);

    const fetchData = async () => {
        try {
            const data = await getAreaConnections();

            setDataList(data);
        } catch (error) {
            console.error('Failed to fetch data:', error);
        }
    };

    const handleSubmit = async () => {
        if (isDisabled) {
            return;
        }

        setLoading(true);

        try {
            await postAreaConnection(areaConnection); // Assuming postAreaConnection is asynchronous
            fetchData();
        } catch (error) {
            console.error('Error submitting data:', error);
        } finally {
            setLoading(false);
        }
    };

    // Handle file selection using PrimeReact FileUpload component
    const handleFileSelect = (event: any) => {
        const uploadedFile = event.files[0];
        setSelectedFile(uploadedFile);
    };

    // Handle upload of transmission file
    const handleTransmissionFileUpload = async () => {
        if (!selectedFile) {
            console.error('No file selected.');
            return;
        }
        setLoading(true);

        await uploadAreaConnectionsFile(selectedFile);
        fetchData();

        setLoading(false);
    };

    return (
        <>
            <LoadingSpinner loading={loading} />
            <Dialog header="Håndtering af transmissionsforbindelser" visible={visible} style={{ width: '50vw' }} onHide={onClose}>
                {/* Container for whole dialog */}
                <div className={`p-fluid ${styles.dialogContainer}`}>
                    <div className={`${styles.formFileInput}`}>
                        <FileUpload
                            mode="basic"
                            id="fileUpload"
                            name="file"
                            accept=".xlsx, .xls"
                            chooseLabel="Choose file"
                            onSelect={handleFileSelect} // Correct event to capture the file
                            auto={false} // Prevent auto-upload
                        />
                    </div>
                </div>
            </Dialog>
        </>
    );
};

export default PredictDialog;
