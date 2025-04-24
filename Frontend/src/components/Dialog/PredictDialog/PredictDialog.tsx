import React, { useState, useEffect } from 'react';
import { Dialog } from 'primereact/dialog';
import styles from './PredictDialog.module.css';
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

    const [selectedFile, setSelectedFile] = useState<File | null>(null);

    const [formVisible, setFormVisible] = useState(false);
    const [loading, setLoading] = useState(false);
    
    


    return (
        <>
            <LoadingSpinner loading={loading} />
            <Dialog header="Håndtering af transmissionsforbindelser" visible={visible} style={{ width: '50vw' }} onHide={onClose}>
                
            </Dialog>
        </>
    );
};

export default PredictDialog;
