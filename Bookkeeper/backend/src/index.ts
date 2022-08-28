import express, { NextFunction } from "express";
import { getAccountNames, getAccount } from "./Accounts_Database_Interface";

const port = 3001;

const app = express();
app.use(express.json());
app.use(
    function (request, response, next: NextFunction) {
        response.setHeader("Access-Control-Allow-Origin", "http://localhost:3000");
        response.setHeader("Access-Control-Allow-Methods", "GET");
        response.setHeader("Access-Control-Allow-Headers", "Content-Type, Access-Control-Allow-Origin, Access-Control-Allow-Methods");
        next();
    }
)

function sendAccount (request: any, responseFromDatabaseServer: any, accountName: string): void {
    getAccount(accountName).then(
        resolvedResponseFromBookkeeperBackend => { responseFromDatabaseServer.status(200).send(resolvedResponseFromBookkeeperBackend); },
        rejectedResponseFromBookkeeperBackend => { responseFromDatabaseServer.status(500).send(rejectedResponseFromBookkeeperBackend); }
    )
}

app.get(
    "/get_account_names",
    function (request, responseFromDatabaseServer) {
        getAccountNames().then(
            resolvedResponseFromBookkeeperBackend => { responseFromDatabaseServer.status(200).send(resolvedResponseFromBookkeeperBackend); },
            rejectedResponseFromBookkeeperBackend => { responseFromDatabaseServer.status(500).send(rejectedResponseFromBookkeeperBackend); }
        );
    }
)

app.get(
    "/get_account",
    (request, responseFromDatabaseServer) => { sendAccount(request, responseFromDatabaseServer, (request.query.name as string)); }
)

app.listen(
    port,
    () => { console.log("Bookkeeper backend is running on port " + port + "."); }
);