import React from 'react';
import { Dialog } from 'primereact/dialog';

interface ConsumptionDialogProps {
    visible: boolean;
    onClose: () => void;
}

const ConsumptionDialog = ({ visible, onClose }: ConsumptionDialogProps) => {
    return (
        <>
            <Dialog
                header="Indlæs Produktionsprofiler"
                visible={visible}
                style={{ width: '50vw' }}
                onHide={onClose}
            >
                <p>Custom content for Indlæs Consumption goes here.</p>
            </Dialog>
        </>
    );
};

export default ConsumptionDialog;
