import Displayer from "./Displayer";

type Props = {
    headerData?: String[]
    bodyData: JSX.Element[][] | String[][]
};

function TableDisplayer (props: Props): JSX.Element {
    const headerCells: JSX.Element[] = [];
    let header: JSX.Element = <></>;
    if ((props.headerData !== null) && (props.headerData !== undefined)) {
        for (let i = 0; i < (props.headerData).length; i++) {
            headerCells.push(
                <th key = { i }>
                    { (props.headerData)[i] }
                </th>
            );
        }
        header = (
            <thead>
                <tr>
                    { headerCells }
                </tr>
            </thead>
        );
    }
    const bodyRows: JSX.Element[] = [];
    for (let i = 0; i < (props.bodyData).length; i++) {
        const dataRow = (props.bodyData)[i];
        const bodyRow: JSX.Element[] = [];
        for (let j = 0; j < dataRow.length; j++) {
            bodyRow.push(
                <td key = { j }>
                    { dataRow[j] }
                </td>
            );
        }
        bodyRows.push(
            <tr key = { i }>
                { bodyRow }
            </tr>
        );
    }
    return (
        <Displayer>
            <table>
                { header }
                <tbody>
                    { bodyRows }
                </tbody>
            </table>
        </Displayer>
    );
};

export default TableDisplayer;