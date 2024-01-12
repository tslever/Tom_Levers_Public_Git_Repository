type Props = {
    children: string | JSX.Element | JSX.Element[]
};

function Displayer (props: Props) {
    return (
        <>
            { props.children }
        </>
    );
};

export default Displayer;