import React from "react";
import Displayer from "./Displayer";

type Props = {
    headerData?: String[]
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
    return (
        <Displayer>
            <table>
                { header }
            </table>
        </Displayer>
    );
};

export default TableDisplayer;