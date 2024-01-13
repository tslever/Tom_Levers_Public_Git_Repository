import Displayer from './Displayer';

type Props = {
  label: string;
};

function ButtonDisplayer(props: Props): JSX.Element {
  const handleButtonClick = async () => {
    try {
      const response = await fetch('http://localhost:8080');
      const data = await response.text();

      // Handle the data or do something with it
      console.log(data);
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };

  return (
    <Displayer>
      <button onClick = { handleButtonClick }>
        { props.label }
      </button>
    </Displayer>
  );
}

export default ButtonDisplayer;