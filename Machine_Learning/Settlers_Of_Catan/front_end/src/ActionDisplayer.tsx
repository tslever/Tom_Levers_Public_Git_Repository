type Props = {
    backgroundColor: string
    children: string
    respond: Function
};

function ActionDisplayer (props: Props) {
    return (
        <div style = { { backgroundColor: props.backgroundColor } } onClick = { () => { props.respond('Player clicked action displayer with child "' + props.children + '".') } }>
            { props.children }
        </div>
    );
};

export default ActionDisplayer;