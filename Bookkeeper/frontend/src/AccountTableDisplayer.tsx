import { CaretakerContext } from './CaretakerContext';
import { useContext } from 'react';
import TableDisplayer from "./TableDisplayer";

function AccountTableDisplayer (): JSX.Element {

    const account = useContext(CaretakerContext);

    return (
        <TableDisplayer
            updateItem = { (dataRow: JSX.Element[] | String[]) => {  } }
            headerData = { ["ID", "Date", "Name", "Account Associated With Value", "Complementary Account", "Value"] }
            bodyData = { account }
        />
    );
};

export default AccountTableDisplayer;