type Props = {
    backgroundColor: string
    children: string | JSX.Element | JSX.Element[]
};

function Displayer (props: Props) {
    return (
        <div style = { { backgroundColor: props.backgroundColor } }>
            { props.children }
        </div>
    );
};

export default Displayer;