type Props = {
    backgroundColor: string
    children: string
    act: Function
};

function ActionDisplayer (props: Props) {
    return (
        <div style = { { backgroundColor: props.backgroundColor } } onClick = { () => { props.act( props.children ) } }>
            { props.children }
        </div>
    );
};

export default ActionDisplayer;