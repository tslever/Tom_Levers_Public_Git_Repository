import { Pool, QueryResult } from "pg";

const pool = new Pool({
    database: "accounts_database",
    host: "localhost",
    password: "password",
    port: 5432,
    user: "postgres"
});

function formatDate(date: Date) {
    return date.getFullYear() + "-" + padToTwoDigits(date.getMonth() + 1) + "-" + padToTwoDigits(date.getDate());
}

export function getAccountNames(): Promise<any> {
    return new Promise<any>(
        function (resolve: (value: any) => void, reject: (reason?: any) => void) {
            pool.query(
                "SELECT id, name FROM accounts",
                function (error: Error, results: QueryResult<any>) {
                    if ((error !== null) && (error !== undefined)) {
                        console.log(error);
                        reject(error);
                    }
                    const accountsAsJsonObjects: any[] = results.rows;
                    const accountNames: string[] = [];
                    for (let accountAsJsonObject of accountsAsJsonObjects) {
                        accountNames.push(accountAsJsonObject.name);
                    }
                    resolve(accountNames);
                }
            );
        }
    );
};

export function getAccount(accountName: string): Promise<any> {
    return new Promise<any>(
        function (resolve: (value: any) => void, reject: (reason?: any) => void) {
            pool.query(
                "SELECT id, date, name, account_associated_with_value, complementary_account, value FROM " + accountName + " ORDER BY Id DESC",
                function (error: Error, results: QueryResult<any>) {
                    if ((error !== null) && (error !== undefined)) {
                        console.log(error);
                        reject(error);
                    }
                    const accountAsJsonObjects: any[] = results.rows;
                    const accountAsArraysOfStrings: string[][] = new Array<Array<string>>();
                    for (let lineItemAsJsonObject of accountAsJsonObjects) {
                        const lineItemAsArrayOfStrings: string[] = [];
                        lineItemAsArrayOfStrings[0] = lineItemAsJsonObject.id;
                        lineItemAsArrayOfStrings[1] = formatDate(lineItemAsJsonObject.date);
                        lineItemAsArrayOfStrings[2] = lineItemAsJsonObject.name;
                        lineItemAsArrayOfStrings[3] = lineItemAsJsonObject.account_associated_with_value;
                        lineItemAsArrayOfStrings[4] = lineItemAsJsonObject.complementary_account;
                        lineItemAsArrayOfStrings[5] = lineItemAsJsonObject.value;
                        accountAsArraysOfStrings.push(lineItemAsArrayOfStrings);
                    }
                    resolve(accountAsArraysOfStrings);
                }
            );
        }
    );
};

function padToTwoDigits(dayOrMonthNumber: number) {
    return dayOrMonthNumber.toString().padStart(2, '0');
}