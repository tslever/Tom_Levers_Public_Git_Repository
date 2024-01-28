import { ReactElement } from "react";

type Props = {
    bodyData: (JSX.Element | number | string)[][],
    colgroup: ReactElement,
    headerData?: String[],
    rowStyles?: { backgroundColor: string }[],
    title?: ReactElement,
    widthPercentage: number
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
        if (props.rowStyles) {
            bodyRows.push(
                <tr key = { i } style = { props.rowStyles[i] }>
                    { bodyRow }
                </tr>
            );
        } else {
            bodyRows.push(
                <tr key = { i }>
                    { bodyRow }
                </tr>
            );
        }
    }
    return (
        <table style = { { width: props.widthPercentage + '%' } }>
            <caption>{ props.title }</caption>
            { props.colgroup }
            { header }
            <tbody>
                { bodyRows }
            </tbody>
        </table>
    );
};

export default TableDisplayer;