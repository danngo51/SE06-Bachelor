export interface ProfileConfRequest {
    excelFile: File;
    profileType: ProfileType;
    shouldOverwriteTables: boolean;
    specificCountriesToUpdate: string;
}

export enum ProfileType {
    DEFAULT,
    CHP,
    HYDRO,
    SOLAR,
    NUCLEAR,
    WIND_ONSHORE,
    WIND_OFFSHORE,
}
