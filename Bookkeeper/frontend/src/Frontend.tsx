import { CaretakerContextProvider } from "./CaretakerContext";
import readResponseStreamToCompletion from "./CommunicationUtils";
import TabDisplayer from "./TabDisplayer";
import TabPaneDisplayer from "./TabPaneDisplayer";
import { useEffect, useState } from 'react';

function Frontend() {

    const [accountNames, setAccountNames] = useState(new Array<string>());
    function fetchReadParseAndSetAccountNames(): void {
        const promiseOfResponse = fetch('http://localhost:3001/get_account_names');
        function parseAndSetAccountNames (accountNamesAsString: string): void {
            const accountNames: string[] = JSON.parse(accountNamesAsString);
            setAccountNames(accountNames);
        }
        promiseOfResponse.then(readResponseStreamToCompletion).then(parseAndSetAccountNames);
    }
    useEffect(() => { fetchReadParseAndSetAccountNames(); }, []);

    const caretakersAndTabDisplayers: JSX.Element[] = [];
    for (let accountName of accountNames) {
        caretakersAndTabDisplayers.push(
            <CaretakerContextProvider
                key = { accountName }
                accountName = { accountName }
            >
                <TabDisplayer />
            </CaretakerContextProvider>
        );
    }

    return (
        <TabPaneDisplayer>
            { caretakersAndTabDisplayers }
        </TabPaneDisplayer>
    );
};

export default Frontend;