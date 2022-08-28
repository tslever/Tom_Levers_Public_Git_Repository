import readResponseStreamToCompletion from "./CommunicationUtils";
import Displayer from "./Displayer";
import { ReactElement } from "react";
import TableDisplayer from "./TableDisplayer";
import TabHeaderDisplayer from "./TabHeaderDisplayer";
import { useEffect, useState } from "react";

type Props = {
    children: ReactElement[]
};

function TabPaneDisplayer (props: Props) {
    const [selectedTabIndex, setSelectedTabIndex] = useState<number>(0);

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

    const tabHeaderDisplayers: JSX.Element[] = [];
    for (let i = 0; i < accountNames.length; i++) {
        tabHeaderDisplayers.push(
            <TabHeaderDisplayer
                title = { accountNames[i] }
                index = { i }
                setSelectedTabIndex = { setSelectedTabIndex }
            />
        )
    }

    return (
        <Displayer>
            <TableDisplayer
                updateItem = { (dataRow: JSX.Element[] | String[]) => { } }
                bodyData = { [tabHeaderDisplayers] }
            />
            { (props.children)[selectedTabIndex] }
        </Displayer>
    );
};

export default TabPaneDisplayer;