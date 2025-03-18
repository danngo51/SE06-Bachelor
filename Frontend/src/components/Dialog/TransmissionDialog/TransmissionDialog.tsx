import React, { useState, useEffect } from 'react';
import { Dialog } from 'primereact/dialog';
import styles from './TransmissionDialog.module.css';
import { areaConnectionDTO } from '../../../data/transmissionConfRequestDTO';
import LoadingSpinner from '../../LoadingSpinner/LoadingSpinner';
import { DataTable } from 'primereact/datatable';
import { Column } from 'primereact/column';
import { Button } from 'primereact/button';
import { InputNumber } from 'primereact/inputnumber';
import { Dropdown } from 'primereact/dropdown';
import { InputText } from 'primereact/inputtext';
import { Calendar } from 'primereact/calendar';
import { FileUpload } from 'primereact/fileupload';

import { getAreaConnections } from '../../../api/api';
import { deleteAreaConnection } from '../../../api/api';
import { uploadAreaConnectionsFile } from '../../../api/api';
import { postAreaConnection } from '../../../api/api';

interface TransmissionDialogProps {
    visible: boolean;
    onClose: () => void;
}

const TransmissionDialog = ({ visible, onClose }: TransmissionDialogProps) => {
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

    //Open close the add transmission part of the dialog
    const toggleForm = () => {
        setFormVisible(!formVisible);
    };

    const handleDeleteAreaConnection = async (id: number) => {
        await deleteAreaConnection(id);

        const updatedDataList = dataList.filter((row) => row.id !== id);
        setDataList(updatedDataList);
    };

    // Button template for delete action
    const deleteButtonTemplate = (rowData: any) => {
        return (
            <Button
                label="Delete"
                icon="pi pi-trash"
                className="p-button-danger"
                onClick={() => handleDeleteAreaConnection(rowData.id)} // Use rowData.id dynamically
                style={{
                    textAlign: 'center',
                    width: '120px',
                }}
            />
        );
    };

    const isEndYearLaterThanStartYear = (startYear: number | null, stopYear: number | null): boolean => {
        if (startYear === null || stopYear === null) {
            return true; // Assume true if one of the years is null
        }
        return stopYear >= startYear;
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
                    {/* Submit Button */}
                    <div className="p-field">
                        <Button
                            label="Upload File"
                            icon="pi pi-upload"
                            className={`p-button-success ${styles.dialogElement}`}
                            onClick={handleTransmissionFileUpload}
                            disabled={!selectedFile} // Disable if no file is selected
                        />
                    </div>
                    <Button
                        icon={formVisible ? 'pi pi-minus' : 'pi pi-plus'}
                        label={formVisible ? 'Annuler tilføjelse' : 'Tilføj transmission'}
                        onClick={toggleForm}
                        className={formVisible ? `${styles.formButton} p-button-secondary` : `${styles.formButton} p-button-info`}
                    />
                    {formVisible && (
                        <form
                            onSubmit={(e) => {
                                e.preventDefault();
                                handleSubmit();
                            }}
                        >
                            {/* Container for form */}
                            <div className={`p-fluid ${styles.formContainer}`}>
                                {/* Area A Dropdown */}
                                <div className="p-field">
                                    <label htmlFor="areaA">Area A</label>
                                    <Dropdown
                                        id="areaA"
                                        value={areaConnection.areaA}
                                        options={Array.from({ length: 246 }, (_, i) => ({
                                            label: `${i}`,
                                            value: i,
                                        }))}
                                        onChange={(e) => {
                                            const value = e.value;
                                            if (value !== areaConnection.areaB) {
                                                setAreaConnection((prev) => ({ ...prev, areaA: value }));
                                            }
                                        }}
                                        placeholder="Select an option"
                                    />
                                </div>

                                {/* Area B Dropdown */}
                                <div className="p-field">
                                    <label htmlFor="areaB">Area B</label>
                                    <Dropdown
                                        id="areaB"
                                        value={areaConnection.areaB}
                                        options={Array.from({ length: 246 }, (_, i) => ({
                                            label: `${i}`,
                                            value: i,
                                        }))}
                                        onChange={(e) => {
                                            const value = e.value;
                                            if (value !== areaConnection.areaA) {
                                                setAreaConnection((prev) => ({ ...prev, areaB: value }));
                                            }
                                        }}
                                        placeholder="Select an option"
                                    />
                                </div>

                                {/* AtoBCapacity InputNumber */}
                                <div className="p-field">
                                    <label htmlFor="AtoBCapacity">AtoBCapacity</label>
                                    <InputNumber
                                        id="AtoBCapacity"
                                        value={areaConnection.atoBCapacity}
                                        onValueChange={(e) =>
                                            setAreaConnection({
                                                ...areaConnection,
                                                atoBCapacity: e.value || 0,
                                            })
                                        }
                                        min={0}
                                        max={100000}
                                        showButtons
                                    />
                                </div>

                                {/* BtoACapacity InputNumber */}
                                <div className="p-field">
                                    <label htmlFor="BtoACapacity">BtoACapacity</label>
                                    <InputNumber
                                        id="BtoACapacity"
                                        value={areaConnection.btoACapacity}
                                        onValueChange={(e) =>
                                            setAreaConnection({
                                                ...areaConnection,
                                                btoACapacity: e.value || 0,
                                            })
                                        }
                                        min={0}
                                        max={100000}
                                        showButtons
                                    />
                                </div>

                                {/* Connection Name Input */}
                                <div className="p-field">
                                    <label htmlFor="connectionName">Connection Name</label>
                                    <InputText
                                        id="connectionName"
                                        value={areaConnection.connectionName}
                                        onChange={(e) =>
                                            setAreaConnection({
                                                ...areaConnection,
                                                connectionName: e.target.value,
                                            })
                                        }
                                        placeholder="Enter Connection Name"
                                    />
                                </div>

                                {/* BtoACapacity InputNumber */}
                                <div className="p-field">
                                    <label htmlFor="startYear">startYear</label>
                                    <InputNumber
                                        id="startYear"
                                        value={areaConnection.startYear}
                                        onValueChange={(e) =>
                                            setAreaConnection({
                                                ...areaConnection,
                                                startYear: e.value || 0,
                                            })
                                        }
                                        min={1900}
                                        max={3000}
                                        useGrouping={false}
                                        showButtons
                                    />
                                </div>

                                {/* BtoACapacity InputNumber */}
                                <div className="p-field">
                                    <label htmlFor="stopYear">stopYear</label>
                                    <InputNumber
                                        id="stopYear"
                                        value={areaConnection.stopYear}
                                        onValueChange={(e) =>
                                            setAreaConnection({
                                                ...areaConnection,
                                                stopYear: e.value || 0,
                                            })
                                        }
                                        min={1900}
                                        max={3000}
                                        useGrouping={false}
                                        showButtons
                                    />
                                </div>

                                <Button disabled={isDisabled} type="submit" label="Indsæt transmission" className="p-button-success" />
                            </div>
                        </form>
                    )}

                    <div className={`p-fluid`}>
                        <h2>List of transmissions:</h2>
                        {/* DataTable rendering */}
                        <div className="p-datatable">
                            <DataTable value={dataList} tableStyle={{ minWidth: '50rem' }}>
                                <Column field="id" header="ID" />
                                <Column field="connectionName" header="Connection name" />
                                <Column field="atoBCapacity" header="A to B Capacity" />
                                <Column field="btoACapacity" header="B to A Capacity" />
                                <Column field="startYear" header="Start date" />
                                <Column field="stopYear" header="End date" />
                                <Column
                                    body={deleteButtonTemplate}
                                    header="Delete connection"
                                    style={{
                                        textAlign: 'center',
                                        width: '150px',
                                    }}
                                />
                            </DataTable>
                        </div>
                    </div>
                </div>
            </Dialog>
        </>
    );
};

export default TransmissionDialog;
