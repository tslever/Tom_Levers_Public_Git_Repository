import { MouseEventHandler } from "react";

type Props = {
    backgroundColor: string
    children: string
    onClick: MouseEventHandler<HTMLDivElement>
};

function ActionDisplayer (props: Props) {
    return (
        <div style = { { backgroundColor: props.backgroundColor } } onClick = { props.onClick }>
            { props.children }
        </div>
    );
};

export default ActionDisplayer;