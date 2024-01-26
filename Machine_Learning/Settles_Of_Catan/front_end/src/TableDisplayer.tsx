import { ReactElement } from "react";

type Props = {
    bodyData: (JSX.Element | string)[][],
    colgroup: ReactElement,
    headerData?: String[],
    widthPercentage: number
};

function TableDisplayer (props: Props): JSX.Element {
    const style = {
        border: '1px solid black'
    }

    const headerCells: JSX.Element[] = [];
    let header: JSX.Element = <></>;
    if ((props.headerData !== null) && (props.headerData !== undefined)) {
        for (let i = 0; i < (props.headerData).length; i++) {
            headerCells.push(
                <th key = { i } style = { style }>
                    { (props.headerData)[i] }
                </th>
            );
        }
        header = (
            <thead style = { style }>
                <tr style = { style }>
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
                <td key = { j } style = { style }>
                    { dataRow[j] }
                </td>
            );
        }
        bodyRows.push(
            <tr key = { i } style = { style }>
                { bodyRow }
            </tr>
        );
    }
    const tableStyle = { border: '1px solid black', width: '100%' }
    return (
        <table style = { tableStyle }>
            { props.colgroup }
            { header }
            <tbody style = { style }>
                { bodyRows }
            </tbody>
        </table>
    );
};

export default TableDisplayer;