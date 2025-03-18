import axios from 'axios';
import { SubdivideRequest } from '../data/subdivideDTO';
import { ProfileConfRequest } from '../data/profileConfRequestDTO';
import { PowerPlantRequest } from '../data/createNewFileDTO';

import { toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { areaConnectionDTO } from '../data/transmissionConfRequestDTO';

// Base URL for your backend API
const API_BASE_URL = import.meta.env.VITE_BACKEND_API_URL || 'http://localhost:8000/';

// #POST /Subdivision
export const subdivideAreas = async (subdivideRequest: SubdivideRequest) => {
    try {
        const response = await axios.post(
            `${API_BASE_URL}/Subdivision`,
            subdivideRequest,
            {
                headers: {
                    'Content-Type': 'application/json',
                },
            }
        );

        toast.success(
            `${subdivideRequest.AreaNames[0]} have been succesfully divided`
        ); // Display success notification
        return response.data;
    } catch (error) {
        toast.error('There was a problem subdividing.'); // Display error notification    throw error;
        throw error;
    }
};

// #POST /SystemPowerPlants/Excel/UploadFile
export const uploadExcelFile = async (file: File): Promise<void> => {
    try {
        const formData = new FormData();
        formData.append('excelFile', file);

        await axios.post(
            `${API_BASE_URL}/SystemPowerPlants/Excel/UploadFile`,
            formData,
            {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            }
        );

        toast.success(
            `File uploaded successfully. New power plants are now inserted into database`
        ); // Display success notification
    } catch (error) {
        toast.error('Error uploading file.'); // Display error notification
        throw error;
    }
};

// #POST /SystemPowerPlants/Excel/CreateNewFile
export const createExcelFile = async (
    powerPlantRequest: PowerPlantRequest
): Promise<void> => {
    try {
        await axios.post(
            `${API_BASE_URL}/SystemPowerPlants/Excel/CreateNewFile`,
            powerPlantRequest,
            {
                headers: {
                    'Content-Type': 'application/json',
                },
            }
        );
        toast.success(`Excel file has been created`); // Display success notification
    } catch (error) {
        toast.error('Error creating new powerplant excel file'); // Display error notification
        throw error;
    }
};

// #POST /UpdateProfile
export const updateProfile = async (formData: FormData): Promise<void> => {
    try {
        await axios.post(`${API_BASE_URL}/UpdateProfile`, formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });

        toast.success(`Profile has been updated`); // Display success notification
    } catch (error) {
        toast.error('Error updating profile'); // Display error notification
        throw error;
    }
};

// #DELETE /SystemPowerPlants/DeleteDatabase
export const deletePowerPlants = async (): Promise<void> => {
    try {
        await axios.delete(
            `${API_BASE_URL}${'/SystemPowerPlants/DeleteDatabase'}`
        );
    } catch (error) {
        toast.error('Error deleting power plants'); // Display error notification
        throw error;
    }
};

// #GET /AreaConnections
export const getAreaConnections = async (): Promise<areaConnectionDTO[]> => {
    try {
        const response = await axios.get(
            `${API_BASE_URL}${'/AreaConnections'}`
        );
        return response.data;
    } catch (error) {
        toast.error('Error retreiving area connections'); // Display error notification
        throw error;
    }
};

// #POST /AreaConnections/AddAreaConnection
export const postAreaConnection = async (
    areaConnectionDTO: areaConnectionDTO
): Promise<void> => {
    try {
        const response = await axios.post(
            `${API_BASE_URL}/AreaConnections/AddAreaConnection`,
            areaConnectionDTO,
            {
                headers: {
                    'Content-Type': 'application/json',
                },
            }
        );
        toast.success(
            `${areaConnectionDTO.areaA} to ${areaConnectionDTO.areaB} have been successfully added`
        );
        return response.data;
    } catch (error) {
        toast.error('There was a problem adding area connection.'); // Display error notification    throw error;
        throw error;
    }
};

// #POST /AreaConnections/Excel/UploadFile
export const uploadAreaConnectionsFile = async (file: File): Promise<void> => {
    try {
        const formData = new FormData();
        formData.append('excelFile', file);

        await axios.post(
            `${API_BASE_URL}/AreaConnections/Excel/UploadFile`,
            formData,
            {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            }
        );
        toast.success(
            `File uploaded successfully. New area connections uploaded.`
        );
    } catch (error) {
        toast.error('Error adding area connections'); // Display error notification
        throw error;
    }
};

// #DELETE /AreaConnections/Delete/{id}
export const deleteAreaConnection = async (id: number): Promise<void> => {
    try {
        await axios.delete(`${API_BASE_URL}/AreaConnections/Delete/${id}`);
        toast.success('Area connection deleted successfully'); // Display success notification
    } catch (error) {
        toast.error('Error deleting area connection'); // Display error notification
        throw error; // Rethrow the error for further handling if needed
    }
};


// #DELETE /AreaConnections/Delete/{id}
export const downloadPMSSettings2 = async (): Promise<void> => {
    try {
        await axios.delete(`${API_BASE_URL}/PMSSettings/Download`);
        toast.success('PMS settings downloaded successfully'); // Display success notification
    } catch (error) {
        toast.error('Error downloading PMS settings'); // Display error notification
        throw error; // Rethrow the error for further handling if needed
    }
};

// #GET /PMSSettings/Download
export const exportPMSSettings = async () => {
    try {
        const response = await axios.get(`${API_BASE_URL}/PMSSettings/Export`, {
            responseType: "blob", // Important for file downloads
        });

        // Create a Blob from the response
        const blob = new Blob([response.data], { type: "text/plain" });
        const url = window.URL.createObjectURL(blob);
        

        // Create a temporary <a> element for downloading
        const link = document.createElement("a");
        link.href = url;
        link.download = "PMS_Settings.txt"; // Set downloaded filename
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

        // Cleanup
        window.URL.revokeObjectURL(url);
    } catch (error) {
        console.error("Error downloading file:", error);
        toast.error("Error downloading file.");
    }
};