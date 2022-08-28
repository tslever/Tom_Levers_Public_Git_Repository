import { createContext, useEffect, useState } from 'react';
import { fetchAccount } from './ServiceLayer';

export const CaretakerContext = createContext(new Array<Array<String>>());

type Props = {
    accountName: string,
    children: JSX.Element | JSX.Element[]
}

export function CaretakerContextProvider (props: Props) {

    const [account, setAccount] = useState(new Array<Array<string>>());
    useEffect(() => {
        const fetchData = async () => {
            const data = await fetchAccount(props.accountName);
            setAccount(data);
        }
        fetchData();
    });

    return (
        <CaretakerContext.Provider value = { account }>
            { props.children }
        </CaretakerContext.Provider>
    );
}