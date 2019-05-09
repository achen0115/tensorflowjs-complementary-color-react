import React, {PureComponent} from 'react';
import * as tf from '@tensorflow/tfjs';
import tinycolor from 'tinycolor2';
import styled from 'styled-components';

import LineChart from './LineChart';

const Container = styled.div`
  max-width: 1127px;
  margin: auto;
  padding: 2rem;
`;

const GraphContainer = styled.div`
  display: flex;
`;

const Input = styled.input`
  margin-right: 1rem;
`;

class App extends PureComponent {
  featuresTensor = null;
  targetsTensor = null;
  model = null;
  size = 10000;
  epochs = 10;
  batchSize = 100;

  constructor(props) {
    super(props);
    this.state = {
      status: '',
      accData: [],
      lossData: [],
      valueR: this._getRandomColorValue(),
      valueG: this._getRandomColorValue(),
      valueB: this._getRandomColorValue(),
      prediction: null,
    };
  }

  componentDidMount() {
    this.setState({status: 'training model'});
    this._createData();
    this._createNetwork();
    this._train();
  }

  handleChange = e => {
    const {name, value} = e.target;
    this.setState({[name]: value});
  };

  render() {
    const {
      status,
      accData,
      lossData,
      valueR,
      valueG,
      valueB,
      prediction,
    } = this.state;

    // const isTrained = status === 'done';
    const isTrained = true;

    const color = [valueR, valueG, valueB];
    const compColor = this._getComplementaryColorArray(color);

    const formatColorString = rgbArr => rgbArr.join(' ');

    const fColor = tinycolor(`rgb (${formatColorString(color)})`);
    const tColor = tinycolor(`rgb (${formatColorString(compColor)})`);
    const pColor = prediction
      ? tinycolor(`rgb (${formatColorString(prediction)})`)
      : prediction;

    return (
      <Container>
        <div style={{padding: '20px 0'}}>
          <h1>STATUS: {status}</h1>

          <GraphContainer>
            {!!accData.length && (
              <div>
                <strong>Accuracy: </strong>
                <LineChart data={accData} dataKey="acc" />
              </div>
            )}
            {!!lossData.length && (
              <div>
                <strong>Loss: </strong>
                <LineChart data={lossData} dataKey="loss" />
              </div>
            )}
          </GraphContainer>

          <h1>Test</h1>
          {isTrained && (
            <div>
              <Input
                onChange={this.handleChange}
                name="valueR"
                value={valueR}
                type="number"
              />
              <Input
                onChange={this.handleChange}
                name="valueG"
                value={valueG}
                type="number"
              />
              <Input
                onChange={this.handleChange}
                name="valueB"
                value={valueB}
                type="number"
              />
              <button onClick={this.resetTestColor}>
                <i className="fas fa-redo" />
              </button>
              <button onClick={this.handlePredict}>Predict</button>
              <table>
                <tr>
                  <th>Feature</th>
                  <th>Target</th>
                  <th>Prediction</th>
                </tr>
                <tr>
                  <td>{this.renderColorCell(fColor)}</td>
                  <td>{this.renderColorCell(tColor)}</td>
                  <td>{!!pColor && this.renderColorCell(pColor)}</td>
                </tr>
              </table>
            </div>
          )}
        </div>
      </Container>
    );
  }

  renderColorCell = color => {
    const colorHexString = color.toHexString();
    return (
      <div>
        <div
          style={{background: colorHexString, width: '100px', height: '50px'}}
        />
        <small>{colorHexString}</small>
      </div>
    );
  };

  resetTestColor = () => {
    this.setState({
      valueR: this._getRandomColorValue(),
      valueG: this._getRandomColorValue(),
      valueB: this._getRandomColorValue(),
    });
  };

  handlePredict = async () => {
    const {valueR, valueG, valueB} = this.state;
    const input = this._normalize([valueR, valueG, valueB]);
    const [r, g, b] = await this.model
      .predict(tf.tensor2d([input], [1, 3]))
      .data();
    this.setState({prediction: this._denormalize([r, g, b])});
  };

  _train = () => {
    const {epochs, batchSize} = this;
    this.model
      .fit(this.featuresTensor, this.targetsTensor, {epochs, batchSize})
      .then(result => {
        const {
          history: {acc, loss},
        } = result;
        this.setState({
          status: 'done',
          accData: acc.map(d => ({acc: d})),
          lossData: loss.map(d => ({loss: d})),
        });
      });
  };

  _createData = () => {
    const features = [];
    const targets = [];

    for (let i = 0; i < this.size; i++) {
      const color = this._getRandomColorArray(); // e.g. [255, 128, 0];
      const compColor = this._getComplementaryColorArray(color); // e.g. [0, 127, 255];

      // normalize value 0 - 255 to float number from 0 - 1
      features.push(this._normalize(color));
      targets.push(this._normalize(compColor));
    }

    /**
     * 2d
     * e.g.
     * [  [25, 100, 150],  [255, 120, 50]  ]
     */
    this.featuresTensor = tf.tensor2d(features);
    this.targetsTensor = tf.tensor2d(targets);
  };

  _createNetwork = () => {
    this.model = tf.sequential();
    // units: the amount of "neurons", or "nodes" the layer has inside it

    this.model.add(tf.layers.dense({inputShape: [3], units: 3}));
    this.model.add(tf.layers.dense({units: 64, useBias: true}));
    this.model.add(tf.layers.dense({units: 32, useBias: true}));
    this.model.add(tf.layers.dense({units: 16, useBias: true}));
    this.model.add(
      tf.layers.dense({units: 3, useBias: true, activation: 'relu'}),
    );

    const optimizer = tf.train.sgd(0.4);
    // setup
    this.model.compile({
      optimizer,
      loss: 'meanSquaredError',
      metrics: ['accuracy'],
    });
  };

  // helper functions

  // return value 0 - 255
  _getRandomColorValue = () => {
    return Math.floor(Math.random() * 256);
  };

  // return an array of rgb values
  _getRandomColorArray = () => {
    return [
      this._getRandomColorValue(),
      this._getRandomColorValue(),
      this._getRandomColorValue(),
    ];
  };

  _getComplementaryColorArray = ([cr, cg, cb]) => {
    const {r, g, b} = tinycolor(`rgb(${cr}, ${cg}, ${cb})`)
      .complement()
      .toRgb();
    return [r, g, b];
  };

  // normalize value 0 - 255 to 0 - 1 float number
  _normalize = color => {
    return color.map(v => v / 255);
  };

  // denormalize 0 - 1 float number to 0 -255
  _denormalize = color => {
    return color.map(v => Math.round(v * 255));
  };
}

export default App;
