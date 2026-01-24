use std::rc::Rc;

use yew::prelude::*;
use yew::{html, Component, Context, Html};
use web_sys::HtmlSelectElement;
use gloo_timers::future::sleep;
use std::time::Duration;
use burn::backend::ndarray::NdArray;

use crate::platform::WasmPlatform;
use crate::ai::{AI, minmax::MinMax, mcts::MCTS, ann::ANNSolo, alphazeutreeko::AlphaZeutreeko};
use crate::logic::{Board, Direction, Pawn, Position, Color};

const SCALING: u32 = 80;

const MARGIN: u32 = 5;
pub enum Msg {
    PawnClick(usize),
    DirectionClick(Direction),
    Restart,
    CreateAi(Color),
    AiShouldPlay,
    AiMoveReady(Option<(usize, Direction)>),
    SetDifficulty(usize),
    SetAiType(usize),
}

enum AiType {
    None,
    MinMax(Color),
    MCTS(Color),
    ANNSolo(Color),
    AlphaZeutreeko(Color),
}

pub struct App {
    board: Board,
    state: Rc<AppState>,
    ai: AiType,
    ai_thinking: bool,
    selected_pawn: Option<usize>,
    difficulty_selected: usize,
    ai_type_selected: usize,
}

impl App {
    fn create_ai(&mut self, color:Color) {
        if self.ai_type_selected == 0 {
            ()
        } else if self.ai_type_selected == 1 {
            self.ai = AiType::MinMax(color);
        } else if self.ai_type_selected == 2 {
            self.ai = AiType::MCTS(color);
        }
        else if self.ai_type_selected == 3 {
            self.ai = AiType::ANNSolo(color);
        }
        else if self.ai_type_selected == 4 {
            self.ai = AiType::AlphaZeutreeko(color);
        }
        else {
            panic!("AI Type not implemented!")
        }
    }
}

#[derive(Clone, PartialEq)]
struct AppState {
    pawn_clicked: Callback<usize>,
    direction_clicked: Callback<Direction>,
}

impl Component for App {
    type Message = Msg;
    type Properties = ();

    fn create(ctx: &Context<Self>) -> Self {
        let pawn_clicked = ctx.link().callback(Msg::PawnClick);
        let direction_clicked = ctx.link().callback(Msg::DirectionClick);
        let state = Rc::new(AppState {
            pawn_clicked,
            direction_clicked,
        });

        let board = Board::default_new();
        Self {
            board,
            state,
            ai: AiType::None,
            ai_thinking: false,
            selected_pawn: None,
            difficulty_selected: 4,
            ai_type_selected: 0,
        }
    }

    fn update(&mut self, ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            Msg::SetAiType(ai_type) => {
                self.ai_type_selected = ai_type;
            }
            Msg::SetDifficulty(difficulty) => {
                self.difficulty_selected = difficulty;
            }
            Msg::PawnClick(pawn_index) => {
                if self.board.next_player == Some(self.board.pawns[pawn_index].color.clone()){
                    self.selected_pawn = Some(pawn_index);
                }
            }
            Msg::DirectionClick(direction) => {
                if let Some(pawn_index) = self.selected_pawn {
                    let mut new_board = self.board.clone();
                    if new_board.move_pawn_until_blocked(pawn_index, &direction) {
                        self.board = new_board;
                        self.selected_pawn = None;

                        ctx.link().send_message(Msg::AiShouldPlay);
                    }
                }
            }
            Msg::AiShouldPlay => {
                match &self.ai {
                    AiType::None => (),                    
                    AiType::MinMax(color) => {
                        if Some(color.clone()) != self.board.next_player {
                            return true;
                        }
                        // Set AI thinking state
                        self.ai_thinking = true;
                        
                        // Spawn async task to calculate AI move
                        let board = self.board.clone();
                        let link = ctx.link().clone();
                        let mut ai: MinMax<WasmPlatform> = MinMax::new(color.clone(), self.difficulty_selected);
                        wasm_bindgen_futures::spawn_local(async move {
                            // Small delay to allow browser to render player's move first
                            sleep(Duration::from_millis(50)).await;
                            let ai_move = ai.ai_play(&board, true);
                            link.send_message(Msg::AiMoveReady(ai_move));
                        });
                    }
                    AiType::MCTS(color) => {
                        if Some(color.clone()) != self.board.next_player {
                            return true;
                        }
                        // Set AI thinking state
                        self.ai_thinking = true;
                        
                        // Spawn async task to calculate AI move
                        let board = self.board.clone();
                        let link = ctx.link().clone();
                        let mut ai: MCTS<WasmPlatform> = MCTS::new(color.clone(), self.difficulty_selected);
                        wasm_bindgen_futures::spawn_local(async move {
                            // Small delay to allow browser to render player's move first
                            sleep(Duration::from_millis(50)).await;
                            let ai_move = ai.ai_play(&board, true);
                            link.send_message(Msg::AiMoveReady(ai_move));
                        });
                    }
                    AiType::ANNSolo(color) => {
                        if Some(color.clone()) != self.board.next_player {
                            return true;
                        }
                        // Set AI thinking state
                        self.ai_thinking = true;
                        
                        // Spawn async task to calculate AI move
                        let board = self.board.clone();
                        let link = ctx.link().clone();
                        let mut ai: ANNSolo<NdArray<f32, i32>, WasmPlatform> = ANNSolo::new(color.clone(), self.difficulty_selected);
                        wasm_bindgen_futures::spawn_local(async move {
                            // Small delay to allow browser to render player's move first
                            sleep(Duration::from_millis(50)).await;
                            let ai_move = ai.ai_play(&board, true);
                            link.send_message(Msg::AiMoveReady(ai_move));
                        });
                    }
                    AiType::AlphaZeutreeko(color) => {
                        if Some(color.clone()) != self.board.next_player {
                            return true;
                        }
                        // Set AI thinking state
                        self.ai_thinking = true;
                        
                        // Spawn async task to calculate AI move
                        let board = self.board.clone();
                        let link = ctx.link().clone();
                        let mut ai: AlphaZeutreeko<NdArray<f32, i32>, WasmPlatform> = AlphaZeutreeko::new(color.clone(), self.difficulty_selected);
                        wasm_bindgen_futures::spawn_local(async move {
                            // Small delay to allow browser to render player's move first
                            sleep(Duration::from_millis(50)).await;
                            let ai_move = ai.ai_play(&board, true);
                            link.send_message(Msg::AiMoveReady(ai_move));
                        });
                    }
                }
            }
            Msg::AiMoveReady(ai_move) => {
                self.ai_thinking = false;
                if let Some((ai_pawn_index, ai_direction)) = ai_move {
                    self.board.move_pawn_until_blocked(ai_pawn_index, &ai_direction);
                }
            }
            Msg::Restart => {
                self.board = Board::default_new();
                self.selected_pawn = None;
                self.ai = AiType::None;
                self.ai_thinking = false;
            }
            Msg::CreateAi(color) => {
                self.create_ai(color);
                ctx.link().send_message(Msg::AiShouldPlay);
            }
        }
        true
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        let app_state = self.state.clone();
        let next_player_text = match self.board.winner() {
            Some(Color::Green) => "Green wins!".to_string(),
            Some(Color::Yellow) => "Yellow wins!".to_string(),
            None => {
                if self.ai_thinking {
                    "🤔 AI is thinking...".to_string()
                } else {
                    match self.board.next_player {
                        Some(Color::Green) => "Green's turn".to_string(),
                        Some(Color::Yellow) => "Yellow's turn".to_string(),
                        None => String::new(),
                    }
                }
            }
        };
        
        // Configuration controls (AI selection, restart)
        let config_view = html! {
            <div class="config-controls">
                <button onclick={ctx.link().callback(|_| Msg::Restart)}>{ "Restart Game" }</button>
                <div class="difficulty-selector">
                    <label>{ "AI Type: " }</label>
                    <select
                        onchange={ctx.link().callback(|e: Event| {
                            let input: HtmlSelectElement = e.target_unchecked_into();
                            Msg::SetAiType(input.value().parse().unwrap_or(0))
                        })}
                    >
                    <option value="0" selected={self.ai_type_selected == 0}>{ "None" }</option>
                    <option value="1" selected={self.ai_type_selected == 1}>{ "MinMax" }</option>
                    <option value="2" selected={self.ai_type_selected == 2}>{ "MCTS" }</option>
                    <option value="3" selected={self.ai_type_selected == 3}>{ "ANN" }</option>
                    <option value="4" selected={self.ai_type_selected == 4}>{ "AlphaZeutreeko" }</option>
                    </select>
                </div>
                <div class="difficulty-selector">
                    <label>{ "AI Difficulty: " }</label>
                    <select
                        onchange={ctx.link().callback(|e: Event| {
                            let input: HtmlSelectElement = e.target_unchecked_into();
                            Msg::SetDifficulty(input.value().parse().unwrap_or(4))
                        })}
                    >
                    <option value="1" selected={self.difficulty_selected == 1}>{ "Very Easy" }</option>
                    <option value="2" selected={self.difficulty_selected == 2}>{ "Easy" }</option>
                    <option value="3" selected={self.difficulty_selected == 3}>{ "Medium" }</option>
                    <option value="4" selected={self.difficulty_selected == 4}>{ "Hard" }</option>
                    <option value="5" selected={self.difficulty_selected == 5}>{ "Very Hard" }</option>
                    <option value="6" selected={self.difficulty_selected == 6}>{ "Expert" }</option>
                    </select>
                </div>
                <button onclick={ctx.link().callback(|_| Msg::CreateAi(Color::Yellow))}>{ "Play against AI as Green" }</button>
                <button onclick={ctx.link().callback(|_| Msg::CreateAi(Color::Green))}>{ "Play against AI as Yellow" }</button>
            </div>
        };

        // Game board and pawns
        let game_view = html! {
            <div class="game-container">
                <BoardView board={self.board.clone()} selected_pawn={self.selected_pawn} />

                // Direction buttons positioned around selected pawn
                {self.render_direction_buttons(ctx)}
            </div>
        };

        html! {
            <ContextProvider<Rc<AppState>> context={app_state}>
                <div class="app-container">
                    {config_view}
                    <h2>{ next_player_text }</h2>
                    {game_view}
                </div>
            </ContextProvider<Rc<AppState>>>
        }
    }
}

impl App {
    fn render_direction_buttons(&self, ctx: &Context<Self>) -> Html {
        if let Some(pawn_index) = self.selected_pawn {
            let pawn = &self.board.pawns[pawn_index];
            let valid_directions = self.board.get_valid_directions(pawn_index);

            let scaling_i32 = SCALING as i32;
            let controls_size = 180;

            let top = 50 + i32::try_from(pawn.position.row).unwrap() * scaling_i32 - (controls_size - scaling_i32) / 2;
            let left = ((f64::from(pawn.position.column as u8) + 0.5 - f64::from(self.board.number_of_columns as u8) / 2.0 ) * f64::from(SCALING)) as i32;

            html! {
                <div class="direction-controls" style={format!("position: relative; top: {}px; left: {}px;", top, left)}>
                    <div class="dir-row">
                        {self.render_direction_button(ctx, Direction::UpLeft, &valid_directions, "↖")}
                        {self.render_direction_button(ctx, Direction::Up, &valid_directions, "↑")}
                        {self.render_direction_button(ctx, Direction::UpRight, &valid_directions, "↗")}
                    </div>
                    <div class="dir-row">
                        {self.render_direction_button(ctx, Direction::Left, &valid_directions, "←")}
                        <div class="dir-spacer"></div>
                        {self.render_direction_button(ctx, Direction::Right, &valid_directions, "→")}
                    </div>
                    <div class="dir-row">
                        {self.render_direction_button(ctx, Direction::DownLeft, &valid_directions, "↙")}
                        {self.render_direction_button(ctx, Direction::Down, &valid_directions, "↓")}
                        {self.render_direction_button(ctx, Direction::DownRight, &valid_directions, "↘")}
                    </div>
                </div>
                }
        } else {
            html! {}
        }
    }

    fn render_direction_button(&self, ctx: &Context<Self>, direction: Direction, valid_directions: &Vec<Direction>, symbol: &str) -> Html {
        let is_valid = valid_directions.contains(&direction);
        if is_valid {
            html! {
                <button
                    class="dir-btn"
                    onclick={ctx.link().callback(move |_| Msg::DirectionClick(direction.clone()))}
                >
                    {symbol}
                </button>
            }
        } else {
            html! {
                <div class="dir-spacer"></div>
            }
        }
    }
}

struct PawnView{
    state: Rc<AppState>,
    _listener: ContextHandle<Rc<AppState>>
}

#[derive(Clone, Properties, PartialEq)]
struct PawnComponent {
    pawn: Pawn,
    position: Position,
    index: usize,
    selected: bool,
}

enum PawnMsg {
    ContextChanged(Rc<AppState>),
}

impl Component for PawnView {
    type Message = PawnMsg;
    type Properties = PawnComponent;
 
    fn create(ctx: &Context<Self>) -> Self {
        let (state, _listener) = ctx
            .link()
            .context::<Rc<AppState>>(ctx.link().callback(PawnMsg::ContextChanged))
            .expect("context to be set");

        Self { state, _listener }
    }

    fn update(&mut self, _ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            PawnMsg::ContextChanged(state) => {
                self.state = state;
                true
            }
        }
    }
 
    fn view(&self, ctx: &Context<Self>) -> Html {
        let my_index = ctx.props().index;
        let onclick = self.state.pawn_clicked.reform(move |_| my_index);

        // Add selection highlight
        let border_style = if ctx.props().selected {
            "border: 3px solid red;"
        } else {
            "border: 2px solid black;"
        };

        html! {
            <div
                onclick={onclick}
                style={format!(
                    "width: {}px; height: {}px; background-color: {}; position: absolute; top: {}px; left: {}px; border-radius: 50%; {}; cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: 18px; font-weight: bold; color: black;",
                    SCALING - MARGIN * 2,
                    SCALING - MARGIN * 2,
                    match ctx.props().pawn.color {
                        Color::Green => "green",
                        Color::Yellow => "yellow",
                    },
                    u32::try_from(ctx.props().position.row).unwrap() * SCALING + MARGIN - 2,
                    u32::try_from(ctx.props().position.column).unwrap() * SCALING + MARGIN - 2,
                    border_style,
                )}
            >
                // {my_index}
            </div>
        }
    }
}

struct BoardView;

#[derive(Clone, Properties, PartialEq)]
struct BoardComponent {
    board: Board,
    selected_pawn: Option<usize>,
}
 
impl Component for BoardView {
    type Message = ();
    type Properties = BoardComponent;

    fn create(_ctx: &Context<Self>) -> Self {
        Self {}
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        let mut pawns = Vec::new();
        for (index, pawn) in ctx.props().board.pawns.iter().enumerate() {
            pawns.push(html! {
                <PawnView
                    pawn={pawn.clone()}
                    position={pawn.position.clone()}
                    index={index}
                    selected={ctx.props().selected_pawn == Some(index)}
                />
            });
        }
        html! {
            <div style={format!(
                "position: absolute; top: {}px; width: {}px; height: {}px; background-image: linear-gradient(0deg, #e0e0e0 1px, transparent 1px), linear-gradient(90deg, #e0e0e0 1px, transparent 1px); background-size: {}px {}px; background-position: 0 0; border: 1px solid #e0e0e0;",
                50,
                SCALING * ctx.props().board.number_of_columns as u32,
                SCALING * ctx.props().board.number_of_rows as u32,
                SCALING, SCALING
            )}>
                {pawns}
            </div>
        }
    }
}
